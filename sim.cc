// Standalone Geant4 simulation of a module (a block of consecutive plates) of the ALLEGRO ECAL barrel in a test-beam setup:
// a beam is fired radially into the module, everything leaving it is counted and killed;
// the back surface is the "virtual detector" of arXiv:2606.05111. No magnetic field.
// The geometry (dimensions, plate structure, materials, layers, readout segmentation) is read at start-up directly
// from the k4geo compact files (allegro_xml.hh) and built as the k4geo driver
// detector/calorimeter/ECalBarrel_NobleLiquid_InclinedTrapezoids_o1_v03_geo.cpp does.
//
//   ./sim run.mac out.root [seed=1] [xmlDir=k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03]
//   ./sim -i [xmlDir]          interactive session with visualisation (runs vis.mac, output in vis.root)
// Size of the simulated module: /module/... commands in the macro, before /run/initialize.
// Physics list: env PHYSLIST (default QGSP_BERT, as in the paper).
#include "allegro_xml.hh"
#include <G4GenericMessenger.hh>
#include <G4Box.hh>
#include <G4ExtrudedSolid.hh>
#include <G4TwoVector.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4NistManager.hh>
#include <G4Material.hh>
#include <G4RunManager.hh>
#include <G4PhysListFactory.hh>
#include <G4VModularPhysicsList.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4GeneralParticleSource.hh>
#include <G4UserRunAction.hh>
#include <G4UserEventAction.hh>
#include <G4UserSteppingAction.hh>
#include <G4AnalysisManager.hh>
#include <G4UImanager.hh>
#include <G4UIExecutive.hh>
#include <G4VisExecutive.hh>
#include <G4Event.hh>
#include <G4Step.hh>
#include <G4Track.hh>
#include <G4SystemOfUnits.hh>
#include <G4PhysicalConstants.hh>
#include <Randomize.hh>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const AllegroXml* X;   // the k4geo description
// simulated module (not from the xml), set with /module/... in the macro
double halfPhiDeg = 20, halfZmm = 1500;   // half width in azimuth of the block of plates, half length along z
bool cryoBack = false;                    // false: virtual detector right behind the liquid, as in the paper

// derived quantities, filled in main() (mm, rad)
double Rmin, Rmax, alpha, dPhi, Lpl, thGrid, thOff, Rback;
int nPlanes, nLay;
std::vector<double> layEnd;            // cumulative layer lengths along the electrode
std::vector<int> mergeTheta, mergeModule;
G4LogicalVolume *bathLV, *worldLV;
int modLo, modHi;   // first and last electrode of the module

// signed distance of p from the plane of the plate starting at azimuth idx*dPhi, and position along that plate
double dist(const G4ThreeVector& p, double idx) {
  double ph = idx * dPhi;
  return -std::sin(ph + alpha) * (p.x() - Rmin * std::cos(ph)) + std::cos(ph + alpha) * (p.y() - Rmin * std::sin(ph));
}
double along(const G4ThreeVector& p, double idx) {
  double ph = idx * dPhi;
  return std::cos(ph + alpha) * (p.x() - Rmin * std::cos(ph)) + std::sin(ph + alpha) * (p.y() - Rmin * std::sin(ph));
}
// which electrode (= gap between two absorbers) and which layer a point belongs to, as the k4geo volumes define them:
// electrodes at i*dPhi, absorbers at (i+1/2)*dPhi, layers = slices along the electrode
bool locate(const G4ThreeVector& p, int& mod, int& lay) {
  double r = p.perp();
  if (r < Rmin - 5 * mm || r > Rmax + 5 * mm) return false;
  double rc = std::min(std::max(r, Rmin), Rmax);
  double s = -Rmin * std::cos(alpha) + std::sqrt(rc * rc - std::pow(Rmin * std::sin(alpha), 2));
  int i = std::lround((p.phi() - std::asin(s * std::sin(alpha) / rc)) / dPhi);   // exact on the electrode plane
  if (dist(p, i - 0.5) < 0) --i;
  else if (dist(p, i + 0.5) >= 0) ++i;
  s = along(p, i);
  if (s < 0 || s > Lpl || i < modLo || i > modHi) return false;
  for (lay = 0; lay < nLay - 1 && s > layEnd[lay];) ++lay;
  mod = ((i % nPlanes) + nPlanes) % nPlanes;
  return true;
}

struct Ev {
  double ebeam, evd, eside, edep, rend;
  std::map<int, double> cells;   // key: layer | module<<4 | theta<<16  -> LAr energy
} ev;
std::vector<int> cLay, cMod, cThe;
std::vector<double> cE, vLayAll, gLayers, gMergeTheta, gMergeModule;

class Det : public G4VUserDetectorConstruction {
  G4GenericMessenger msg{this, "/module/", "simulated module"};
  G4Material* mat(const std::string& role) {
    auto name = X->role.at(role);
    if (auto m = G4Material::GetMaterial(name, false)) return m;
    auto& d = X->mat.at(name);
    auto m = new G4Material(name, d.density * g / cm3, d.comp.size());
    for (auto& [el, n] : d.comp) {
      auto e = G4NistManager::Instance()->FindOrBuildElement(el);
      if (d.fractions) m->AddElement(e, n);
      else m->AddElement(e, int(std::lround(n)));
    }
    return m;
  }
public:
  Det() {
    msg.DeclareProperty("halfPhiDeg", halfPhiDeg, "half width in azimuth of the block of plates [deg]");
    msg.DeclareProperty("halfZ", halfZmm, "half length of the module along z [mm]");
    msg.DeclareProperty("cryoBack", cryoBack, "keep the back cryostat wall in front of the virtual detector");
  }
  G4VPhysicalVolume* Construct() override {
    double halfPhi = std::min(halfPhiDeg, 60.) * deg;
    double dz = std::min(halfZmm, X->dz) * mm;
    double c1 = X->cryoRmin1, c2 = X->cryoRmin2, C1 = X->cryoRmax1, C2 = X->cryoRmax2;
    double tIn = X->tInner, tGlue = X->tGlue, tOut = X->tOuter, tRO = X->tReadout;

    worldLV = new G4LogicalVolume(new G4Box("World", C2 + 1 * m, C2 + 1 * m, dz + 1 * m),
                                  G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic"), "World");
    auto world = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);
    // The module is a block of consecutive plates, like a supermodule: electrodes modLo..modHi with an absorber on
    // each side. Its sides follow the inclined plates (planes of the "plates" modLo-1 and modHi+1), front and back
    // are at constant radius. The block is shifted by half the azimuth swept by a plate, so that a radial beam at
    // phi = 0 crosses its centre at mid depth.
    double dMax = std::asin(Lpl * std::sin(alpha) / Rmax);
    modLo = std::ceil((-halfPhi - dMax / 2) / dPhi);
    modHi = std::floor((halfPhi - dMax / 2) / dPhi);
    auto pt = [&](double idx, double r) {   // point at radius r on the (extended) plane of plate idx
      double sp = -Rmin * std::cos(alpha) + std::sqrt(r * r - std::pow(Rmin * std::sin(alpha), 2)), ph = idx * dPhi;
      return G4TwoVector(Rmin * std::cos(ph) + sp * std::cos(ph + alpha), Rmin * std::sin(ph) + sp * std::sin(ph + alpha));
    };
    auto arc = [&](std::vector<G4TwoVector>& v, double r, double ia, double ib) {   // chords of at most 3 deg
      double pa = pt(ia, r).phi(), pb = pt(ib, r).phi();
      int n = std::max(1, int(std::ceil(std::abs(pb - pa) / (3 * deg))));
      for (int k = 0; k <= n; ++k) { double ph = pa + (pb - pa) * k / n; v.emplace_back(r * std::cos(ph), r * std::sin(ph)); }
    };
    auto band = [&](const char* n, double r0, double r1, G4Material* m) {   // radial slice r0..r1 of the module
      std::vector<G4TwoVector> v;
      arc(v, r1, modLo - 1., modHi + 1.);
      arc(v, r0, modHi + 1., modLo - 1.);
      auto lv = new G4LogicalVolume(new G4ExtrudedSolid(n, v, dz, G4TwoVector(), 1., G4TwoVector(), 1.), m, n);
      new G4PVPlacement(nullptr, {}, lv, n, worldLV, false, 0, true);
      return lv;
    };
    band("CryoFront", c1, c2, mat("cryo"));
    bathLV = band("Bath", c2, C1, mat("active"));   // services + bath + gaps: all liquid
    if (cryoBack) band("CryoBack", C1, C2, mat("cryo"));
    Rback = cryoBack ? C2 : C1;

    // absorber: inner core (a different material in layer 0), glue and outer skin on both faces
    auto box = [&](const char* n, double t, double len, G4Material* m) {
      return new G4LogicalVolume(new G4Box(n, t / 2, dz, len / 2), m, n);
    };
    auto absLV = box("Outer", tIn + tGlue + tOut, Lpl, mat("outer"));
    auto glueLV = box("Glue", tIn + tGlue, Lpl, mat("glue"));
    new G4PVPlacement(nullptr, {}, glueLV, "Glue", absLV, false, 0);
    double l0 = layEnd[0];
    new G4PVPlacement(nullptr, {0, 0, -Lpl / 2 + l0 / 2}, box("Core0", tIn, l0, mat("core0")), "Core0", glueLV, false, 0);
    new G4PVPlacement(nullptr, {0, 0, l0 / 2}, box("Core", tIn, Lpl - l0, mat("core")), "Core", glueLV, false, 0);
    auto roLV = box("Readout", tRO, Lpl, mat("readout"));

    // plates start at Rmin at azimuth phi and run for a length Lpl at an angle alpha to the radial direction
    auto place = [&](double idx, G4LogicalVolume* lv, const char* n, int copy) {
      double phi = idx * dPhi;
      G4ThreeVector u(std::cos(phi + alpha), std::sin(phi + alpha), 0), z(0, 0, 1);
      G4RotationMatrix rot(z.cross(u), z, u);   // local x = plate normal, y = barrel z, z = along plate
      G4ThreeVector c = Rmin * G4ThreeVector(std::cos(phi), std::sin(phi), 0) + 0.5 * Lpl * u;
      new G4PVPlacement(G4Transform3D(rot, c), lv, n, bathLV, false, copy, copy < modLo + 2 || copy > modHi - 1);
    };
    for (int i = modLo; i <= modHi + 1; ++i) {   // electrode at i*dPhi, absorbers at (i -+ 1/2)*dPhi
      place(i - 0.5, absLV, "Outer", i);
      if (i <= modHi) place(i, roLV, "Readout", i);
    }
    return world;
  }
};

class Gun : public G4VUserPrimaryGeneratorAction {
  G4GeneralParticleSource gps;   // fully configured from the macro
public:
  void GeneratePrimaries(G4Event* e) override { gps.GeneratePrimaryVertex(e); }
};

class Run : public G4UserRunAction {
  G4String out;
public:
  explicit Run(G4String o) : out(o) {
    auto am = G4AnalysisManager::Instance();
    am->CreateNtuple("events", "ALLEGRO ECAL barrel");
    for (auto n : {"Ebeam", "Evd", "Eside", "Edep", "Rend"}) am->CreateNtupleDColumn(n);   // MeV, mm
    am->CreateNtupleDColumn("Elayer_all", vLayAll);   // all deposits (LAr + passive) per layer
    am->CreateNtupleIColumn("cell_layer", cLay);
    am->CreateNtupleIColumn("cell_module", cMod);     // first module of the merged group
    am->CreateNtupleIColumn("cell_theta", cThe);      // first theta bin of the merged group
    am->CreateNtupleDColumn("cell_E", cE);            // LAr deposit, uncalibrated
    am->FinishNtuple();
    am->CreateNtuple("geo", "geometry used, one row");   // read by ntuple.py to place the cells
    for (auto n : {"rmin", "rmax", "angle", "nplanes", "theta_grid", "theta_offset", "half_phi_deg", "half_z", "cryo_back"}) am->CreateNtupleDColumn(n);
    am->CreateNtupleDColumn("layers", gLayers);
    am->CreateNtupleDColumn("merge_theta", gMergeTheta);
    am->CreateNtupleDColumn("merge_module", gMergeModule);
    am->FinishNtuple();
  }
  void BeginOfRunAction(const G4Run*) override { G4AnalysisManager::Instance()->OpenFile(out); }
  void EndOfRunAction(const G4Run*) override {
    auto am = G4AnalysisManager::Instance();
    int c = 0;
    for (double v : {Rmin, Rmax, alpha, double(nPlanes), thGrid, thOff, halfPhiDeg, halfZmm, double(cryoBack)}) am->FillNtupleDColumn(1, c++, v);
    am->AddNtupleRow(1);
    am->Write(); am->CloseFile();
  }
};

class Evt : public G4UserEventAction {
public:
  void BeginOfEventAction(const G4Event* e) override {
    ev = Ev{};
    ev.rend = -1;
    ev.ebeam = e->GetPrimaryVertex()->GetPrimary()->GetTotalEnergy();
    vLayAll.assign(nLay, 0.);
  }
  void EndOfEventAction(const G4Event*) override {
    cLay.clear(); cMod.clear(); cThe.clear(); cE.clear();
    for (auto& [k, e] : ev.cells) { cLay.push_back(k & 15); cMod.push_back((k >> 4) & 4095); cThe.push_back(k >> 16); cE.push_back(e); }
    auto am = G4AnalysisManager::Instance();
    int c = 0;
    for (double v : {ev.ebeam, ev.evd, ev.eside, ev.edep, ev.rend}) am->FillNtupleDColumn(c++, v);
    am->AddNtupleRow();
  }
};

class Step : public G4UserSteppingAction {
public:
  void UserSteppingAction(const G4Step* s) override {
    auto tr = s->GetTrack();
    auto pre = s->GetPreStepPoint(), post = s->GetPostStepPoint();
    auto preLV = pre->GetPhysicalVolume()->GetLogicalVolume();
    double e = s->GetTotalEnergyDeposit();
    if (e > 0) {
      G4ThreeVector p = 0.5 * (pre->GetPosition() + post->GetPosition());
      int mod, l;
      ev.edep += e;
      if (locate(p, mod, l)) {
        vLayAll[l] += e;
        if (preLV == bathLV) {   // liquid between two absorbers = active
          int the = std::lround((std::atan2(p.perp(), p.z()) - thOff) / thGrid);
          mod -= mod % mergeModule[l];
          the -= the % mergeTheta[l];
          ev.cells[l | (mod << 4) | (the << 16)] += e;
        }
      }
    }
    // anything leaving the calorimeter is counted and killed; the outer surface is the virtual detector
    auto postPV = post->GetPhysicalVolume();
    if (preLV != worldLV && (!postPV || postPV->GetLogicalVolume() == worldLV)) {
      auto d = tr->GetDefinition();
      int b = d->GetBaryonNumber();   // baryons: kinetic energy; antibaryons: + annihilation; others: total energy
      double esc = b > 0 ? tr->GetKineticEnergy() : b < 0 ? tr->GetKineticEnergy() + 2 * d->GetPDGMass() : tr->GetTotalEnergy();
      (post->GetPosition().perp() > Rback - 2 * mm ? ev.evd : ev.eside) += esc;   // back face = chords of the circle Rback
      tr->SetTrackStatus(fStopAndKill);
    }
    if (tr->GetTrackID() == 1 && tr->GetTrackStatus() != fAlive && ev.rend < 0) ev.rend = post->GetPosition().perp();
  }
};
}  // namespace

int main(int argc, char** argv) {
  bool inter = argc > 1 && std::string(argv[1]) == "-i";
  if (!inter && argc < 3) { G4cerr << "usage: sim run.mac out.root [seed=1] [xmlDir]\n       sim -i [xmlDir]" << G4endl; return 1; }
  auto ui = inter ? new G4UIExecutive(argc, argv) : nullptr;
  G4Random::setTheSeed(!inter && argc > 3 ? std::atol(argv[3]) : 1);
  const char* xmlDir = inter ? (argc > 2 ? argv[2] : nullptr) : (argc > 4 ? argv[4] : nullptr);
  X = new AllegroXml(xmlDir ? xmlDir : "k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03");
  Rmin = X->rmin; Rmax = X->rmax; alpha = X->angle; nPlanes = X->nPlanes; dPhi = twopi / nPlanes;
  Lpl = -Rmin * std::cos(alpha) + std::sqrt(Rmax * Rmax - std::pow(Rmin * std::sin(alpha), 2));   // planeLength
  thGrid = X->thetaGrid; thOff = X->thetaOffset;
  nLay = X->layers.size(); mergeTheta = X->mergeTheta; mergeModule = X->mergeModule;
  for (int l = 0; l < nLay; ++l) layEnd.push_back((l ? layEnd[l - 1] : 0) + X->layers[l]);
  gLayers = X->layers; gMergeTheta.assign(mergeTheta.begin(), mergeTheta.end()); gMergeModule.assign(mergeModule.begin(), mergeModule.end());
  if (std::abs(layEnd.back() - Lpl) > 0.5 * mm || nLay > 16 || nPlanes > 4096) throw std::runtime_error("inconsistent layers / planes in the xml");

  // ponytail: sequential run manager, event data in file-scope globals; parallelise with one job per seed
  auto rm = new G4RunManager;
  const char* pl = std::getenv("PHYSLIST");
  rm->SetUserInitialization(G4PhysListFactory().GetReferencePhysList(pl ? pl : "QGSP_BERT"));
  rm->SetUserInitialization(new Det);
  rm->SetUserAction(new Gun);
  rm->SetUserAction(new Run(inter ? "vis.root" : argv[2]));
  rm->SetUserAction(new Evt);
  rm->SetUserAction(new Step);
  if (inter) {
    G4VisExecutive vis;
    vis.Initialize();
    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute vis.mac");
    ui->SessionStart();
    delete ui;
  } else {
    G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  }
  delete rm;
}
