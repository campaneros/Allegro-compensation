// Standalone Geant4 simulation of a module (sector) of the ALLEGRO ECAL barrel in a test-beam setup:
// a beam is fired radially into the module, everything leaving it is counted and killed;
// the back surface is the "virtual detector" of arXiv:2606.05111. No magnetic field.
// The whole geometry (dimensions, plate structure, materials, layers, readout segmentation) is read from
// geometry.txt, produced by xml2geo.py from the k4geo compact files. The construction follows
// k4geo detector/calorimeter/ECalBarrel_NobleLiquid_InclinedTrapezoids_o1_v03_geo.cpp.
//
//   ./sim run.mac out.root [seed=1] [geometry.txt]
// Physics list: env PHYSLIST (default QGSP_BERT, as in the paper).
#include <G4Box.hh>
#include <G4Tubs.hh>
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
std::map<std::string, std::vector<std::string>> kv, matDef;   // geometry.txt
double num(const std::string& k, size_t i = 0) {
  if (!kv.count(k) || kv[k].size() <= i) throw std::runtime_error("geometry file: missing " + k);
  return std::stod(kv[k][i]);
}
void readGeo(const std::string& fn) {
  std::ifstream f(fn);
  if (!f) throw std::runtime_error("cannot open " + fn + " (create it with xml2geo.py)");
  for (std::string line; std::getline(f, line);) {
    std::istringstream is(line.substr(0, line.find('#')));
    std::string key, tok;
    if (!(is >> key)) continue;
    bool isMat = key == "material";
    if (isMat) is >> key;
    auto& v = isMat ? matDef[key] : kv[key];
    while (is >> tok) v.push_back(tok);
  }
}

// geometry, filled in main() from the file (mm, rad)
double Rmin, Rmax, alpha, dPhi, Lpl, thGrid, thOff, Rback;
int nPlanes, nLay;
std::vector<double> layEnd;            // cumulative layer lengths along the electrode
std::vector<int> mergeTheta, mergeModule;
G4LogicalVolume *bathLV, *worldLV;

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
  if (s < 0 || s > Lpl) return false;
  for (lay = 0; lay < nLay - 1 && s > layEnd[lay];) ++lay;
  mod = ((i % nPlanes) + nPlanes) % nPlanes;
  return true;
}

struct Ev {
  double ebeam, evd, eside, edep, rend;
  std::map<int, double> cells;   // key: layer | module<<4 | theta<<16  -> LAr energy
} ev;
std::vector<int> cLay, cMod, cThe;
std::vector<double> cE, vLayAll;

class Det : public G4VUserDetectorConstruction {
  G4Material* mat(const std::string& role) {
    auto name = kv.at("mat_" + role).at(0);
    if (auto m = G4Material::GetMaterial(name, false)) return m;
    auto& t = matDef.at(name);   // density, natoms|fraction, (element, amount)...
    int n = (t.size() - 2) / 2;
    auto m = new G4Material(name, std::stod(t[0]) * g / cm3, n);
    for (int i = 0; i < n; ++i) {
      auto el = G4NistManager::Instance()->FindOrBuildElement(t[2 + 2 * i]);
      if (t[1] == "natoms") m->AddElement(el, std::stoi(t[3 + 2 * i]));
      else m->AddElement(el, std::stod(t[3 + 2 * i]));
    }
    return m;
  }
public:
  G4VPhysicalVolume* Construct() override {
    // simulated module: a sector of the barrel, +-halfPhi in azimuth and +-dz along the beam line
    double halfPhi = std::min(num("sector_half_deg"), 180.) * deg;
    double dz = std::min(num("z_half"), num("dz")) * mm, cdz = dz;
    double c1 = num("cryo_rmin1") * mm, c2 = num("cryo_rmin2") * mm, C1 = num("cryo_rmax1") * mm, C2 = num("cryo_rmax2") * mm;
    double tIn = num("t_inner") * mm, tGlue = num("t_glue") * mm, tOut = num("t_outer") * mm, tRO = num("t_readout") * mm;

    worldLV = new G4LogicalVolume(new G4Box("World", C2 + 1 * m, C2 + 1 * m, cdz + 1 * m),
                                  G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic"), "World");
    auto world = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);
    auto tubs = [&](const char* n, double r0, double r1, double hz, G4Material* m) {
      auto lv = new G4LogicalVolume(new G4Tubs(n, r0, r1, hz, -halfPhi, 2 * halfPhi), m, n);
      new G4PVPlacement(nullptr, {}, lv, n, worldLV, false, 0, true);
      return lv;
    };
    tubs("CryoFront", c1, c2, cdz, mat("cryo"));
    bathLV = tubs("Bath", c2, C1, dz, mat("active"));   // services + bath + gaps: all liquid
    bool back = num("use_cryo_back") != 0;
    if (back) tubs("CryoBack", C1, C2, cdz, mat("cryo"));
    Rback = back ? C2 : C1;

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
    // keep only the plates fully inside the sector (a plate sweeps dMax in azimuth); the whole ring if halfPhi = 180 deg
    double dMax = std::asin(Lpl * std::sin(alpha) / Rmax), margin = 0.3 * deg;
    bool ring = halfPhi >= 180 * deg;
    int i0 = ring ? 0 : std::ceil((-halfPhi + margin) / dPhi), i1 = ring ? nPlanes - 1 : std::floor((halfPhi - dMax - margin) / dPhi);
    for (int i = i0; i <= i1; ++i)
      for (int isAbs = 0; isAbs < 2; ++isAbs) {
        double phi = (i + 0.5 * isAbs) * dPhi;
        G4ThreeVector u(std::cos(phi + alpha), std::sin(phi + alpha), 0), z(0, 0, 1);
        G4RotationMatrix rot(z.cross(u), z, u);   // local x = plate normal, y = barrel z, z = along plate
        G4ThreeVector c = Rmin * G4ThreeVector(std::cos(phi), std::sin(phi), 0) + 0.5 * Lpl * u;
        new G4PVPlacement(G4Transform3D(rot, c), isAbs ? absLV : roLV, isAbs ? "Outer" : "Readout", bathLV, false, i, i < i0 + 2);
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
  }
  void BeginOfRunAction(const G4Run*) override { G4AnalysisManager::Instance()->OpenFile(out); }
  void EndOfRunAction(const G4Run*) override { auto am = G4AnalysisManager::Instance(); am->Write(); am->CloseFile(); }
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
      (post->GetPosition().perp() > Rback - 0.1 * mm ? ev.evd : ev.eside) += esc;
      tr->SetTrackStatus(fStopAndKill);
    }
    if (tr->GetTrackID() == 1 && tr->GetTrackStatus() != fAlive && ev.rend < 0) ev.rend = post->GetPosition().perp();
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { G4cerr << "usage: sim run.mac out.root [seed=1] [geometry.txt]" << G4endl; return 1; }
  G4Random::setTheSeed(argc > 3 ? std::atol(argv[3]) : 1);
  readGeo(argc > 4 ? argv[4] : "geometry.txt");
  Rmin = num("rmin") * mm; Rmax = num("rmax") * mm; alpha = num("angle");
  nPlanes = int(num("nplanes")); dPhi = twopi / nPlanes;
  Lpl = -Rmin * std::cos(alpha) + std::sqrt(Rmax * Rmax - std::pow(Rmin * std::sin(alpha), 2));   // planeLength
  thGrid = num("theta_grid"); thOff = num("theta_offset");
  nLay = kv.at("layers").size();
  for (int l = 0; l < nLay; ++l) {
    layEnd.push_back((l ? layEnd[l - 1] : 0) + num("layers", l) * mm);
    mergeTheta.push_back(int(num("merge_theta", l)));
    mergeModule.push_back(int(num("merge_module", l)));
  }
  if (std::abs(layEnd.back() - Lpl) > 0.5 * mm || nLay > 16 || nPlanes > 4096) throw std::runtime_error("geometry file: inconsistent layers / planes");

  // ponytail: sequential run manager, event data in file-scope globals; parallelise with one job per seed
  auto rm = new G4RunManager;
  const char* pl = std::getenv("PHYSLIST");
  rm->SetUserInitialization(G4PhysListFactory().GetReferencePhysList(pl ? pl : "QGSP_BERT"));
  rm->SetUserInitialization(new Det);
  rm->SetUserAction(new Gun);
  rm->SetUserAction(new Run(argv[2]));
  rm->SetUserAction(new Evt);
  rm->SetUserAction(new Step);
  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  delete rm;
}
