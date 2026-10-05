// Standalone Geant4 model of a phi-sector of the ALLEGRO ECAL barrel, used as a test-beam-like module:
// beam fired radially into the inclined Pb/LAr stack, everything leaving the module is counted and killed
// (back face = "virtual detector" of arXiv:2606.05111).
// All numbers from k4geo FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/{ECalBarrel_thetamodulemerged,DectDimensions,materials}.xml
//
//   ./sim run.mac out.root [seed=1] [backAl_mm=0]
// backAl_mm: aluminium between the LAr bath and the virtual detector (102.7 = back cryostat wall + solenoid).
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
#include <map>
#include <vector>

namespace {
// ---- geometry constants (k4geo) ----
const double Rmin = 2172.8 * mm;   // EMBarrel_rmin = BarECal_rmin(2100) + air(49) + CryoBarrelFront(13.8) + bath(10)
const double Rmax = 2578.3 * mm;   // EMBarrel_rmax = BarECal_rmax(2770) - air(49) - CryoBarrelBack(102.7) - bath(40)
const double alpha = 50.18 * deg;  // InclinationAngle
const double dPhi = twopi / 1536;  // ECalBarrelNumPlanes
const double L = -Rmin * std::cos(alpha) + std::sqrt(Rmax * Rmax - std::pow(Rmin * std::sin(alpha), 2));  // planeLength
const int nLay = 11;
const double layLen[nLay] = {2.33596, 4.75685, 4.89843, 5.04000, 5.20989, 5.36562, 5.54966, 5.73371, 5.94607, 6.15843, 6.39909};  // cm, along the electrode
const double tPb = 1.8 * mm, tGlue = 0.1 * mm, tSteel = 0.1 * mm, tPCB = 1.2 * mm;
const double bathFront = 10 * mm, bathBack = 40 * mm, cryoFront = 13.8 * mm;
const double thGrid = 0.009817477 / 4, thOff = 0.5902785;       // theta segmentation
const int mergeTheta[nLay] = {4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4}, mergeModule = 2;
// ---- size of the simulated sector (not from k4geo) ----
const double halfPhi = 22 * deg, halfZ = 500 * mm;

double layR[nLay + 1];        // layer boundaries are surfaces of constant radius
double Rback;                 // radius of the back face = virtual detector
G4LogicalVolume *bathLV, *worldLV;

double sOfR(double r) { return -Rmin * std::cos(alpha) + std::sqrt(r * r - std::pow(Rmin * std::sin(alpha), 2)); }  // position along the electrode
double rOfS(double s) { return std::sqrt(Rmin * Rmin + s * s + 2 * Rmin * s * std::cos(alpha)); }

struct Ev {
  double ebeam, evd, eside, edep, rend, eLayAll[nLay];
  std::map<int, double> cells;   // key: layer | (module+2048)<<4 | theta<<16  -> LAr energy
} ev;
std::vector<int> cLay, cMod, cThe;
std::vector<double> cE, vLayAll;

class Det : public G4VUserDetectorConstruction {
  double backAl;
public:
  explicit Det(double b) : backAl(b) {}
  G4VPhysicalVolume* Construct() override {
    auto nist = G4NistManager::Instance();
    auto el = [&](const char* n) { return nist->FindOrBuildElement(n); };
    auto lar = new G4Material("LAr", 1.396 * g / cm3, 1); lar->AddElement(el("Ar"), 1);
    auto pcb = new G4Material("PCB", 1.7 * g / cm3, 5);
    pcb->AddElement(el("Si"), 0.180774); pcb->AddElement(el("O"), 0.405633); pcb->AddElement(el("C"), 0.278042);
    pcb->AddElement(el("H"), 0.0684428); pcb->AddElement(el("Br"), 0.0671091);
    auto glue = new G4Material("lArCaloGlue", 1.69 * g / cm3, 4);
    glue->AddElement(el("C"), 5); glue->AddElement(el("H"), 8); glue->AddElement(el("O"), 4); glue->AddElement(el("Si"), 1);
    auto steel = new G4Material("lArCaloSteel", 7.84 * g / cm3, 3);
    steel->AddElement(el("Fe"), 0.7175); steel->AddElement(el("Cr"), 0.19); steel->AddElement(el("Ni"), 0.0925);
    auto pb = nist->FindOrBuildMaterial("G4_Pb"), al = nist->FindOrBuildMaterial("G4_Al");

    worldLV = new G4LogicalVolume(new G4Box("World", 3 * m, 3 * m, 3 * m), nist->FindOrBuildMaterial("G4_Galactic"), "World");
    auto world = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);
    auto tubs = [&](const char* n, double r0, double r1, G4Material* mat) {
      auto lv = new G4LogicalVolume(new G4Tubs(n, r0, r1, halfZ, -halfPhi, 2 * halfPhi), mat, n);
      new G4PVPlacement(nullptr, {}, lv, n, worldLV, false, 0, true);
      return lv;
    };
    tubs("CryoFront", Rmin - bathFront - cryoFront, Rmin - bathFront, al);
    bathLV = tubs("Bath", Rmin - bathFront, Rmax + bathBack, lar);
    Rback = Rmax + bathBack + backAl;
    if (backAl > 0) tubs("CryoBack", Rmax + bathBack, Rback, al);

    // absorber plate: Pb core, glue, steel skin; in layer 0 the core is (inactive) LAr instead of Pb
    auto box = [&](const char* n, double t, double len, G4Material* mat) {
      return new G4LogicalVolume(new G4Box(n, t / 2, halfZ, len / 2), mat, n);
    };
    auto absLV = box("Steel", tPb + tGlue + tSteel, L, steel);
    auto glueLV = box("Glue", tPb + tGlue, L, glue);
    new G4PVPlacement(nullptr, {}, glueLV, "Glue", absLV, false, 0);
    double l0 = layLen[0] * cm;
    new G4PVPlacement(nullptr, {0, 0, -L / 2 + l0 / 2}, box("CoreLAr", tPb, l0, lar), "CoreLAr", glueLV, false, 0);
    new G4PVPlacement(nullptr, {0, 0, l0 / 2}, box("Pb", tPb, L - l0, pb), "Pb", glueLV, false, 0);
    auto pcbLV = box("PCB", tPCB, L, pcb);

    // plates start at Rmin at azimuth phi and run for a length L at an angle alpha to the radial direction
    double dMax = std::asin(L * std::sin(alpha) / Rmax);                  // azimuth swept by one plate
    int i0 = std::ceil((-halfPhi + 0.3 * deg) / dPhi), i1 = std::floor((halfPhi - dMax - 0.3 * deg) / dPhi);
    for (int i = i0; i <= i1; ++i)
      for (int isAbs = 0; isAbs < 2; ++isAbs) {                           // electrode at i*dPhi, absorber at (i+1/2)*dPhi
        double phi = (i + 0.5 * isAbs) * dPhi;
        G4ThreeVector u(std::cos(phi + alpha), std::sin(phi + alpha), 0), z(0, 0, 1);
        G4RotationMatrix rot(z.cross(u), z, u);                           // local x = plate normal, y = barrel z, z = along plate
        G4ThreeVector c = Rmin * G4ThreeVector(std::cos(phi), std::sin(phi), 0) + 0.5 * L * u;
        new G4PVPlacement(G4Transform3D(rot, c), isAbs ? absLV : pcbLV, isAbs ? "Steel" : "PCB", bathLV, false, i, i == i0);
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
    am->CreateNtuple("events", "ALLEGRO ECAL sector");
    for (auto n : {"Ebeam", "Evd", "Eside", "Edep", "Rend"}) am->CreateNtupleDColumn(n);   // MeV, mm
    am->CreateNtupleDColumn("Elayer_all", vLayAll);   // all deposits (LAr + passive) per layer
    am->CreateNtupleIColumn("cell_layer", cLay);
    am->CreateNtupleIColumn("cell_module", cMod);     // merged module index (azimuth)
    am->CreateNtupleIColumn("cell_theta", cThe);      // merged theta bin
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
  }
  void EndOfEventAction(const G4Event*) override {
    cLay.clear(); cMod.clear(); cThe.clear(); cE.clear();
    for (auto& [k, e] : ev.cells) { cLay.push_back(k & 15); cMod.push_back(((k >> 4) & 4095) - 2048); cThe.push_back(k >> 16); cE.push_back(e); }
    vLayAll.assign(ev.eLayAll, ev.eLayAll + nLay);
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
      double r = p.perp();
      ev.edep += e;
      if (r > Rmin && r < Rmax) {
        int l = 0;
        while (l < nLay - 1 && r > layR[l + 1]) ++l;
        ev.eLayAll[l] += e;
        if (preLV == bathLV) {   // LAr between the plates = active
          double sp = sOfR(r);
          int mod = std::lround((p.phi() - std::asin(sp * std::sin(alpha) / r)) / dPhi);   // nearest electrode
          int the = std::lround((std::atan2(r, p.z()) - thOff) / thGrid);
          mod = std::floor(double(mod) / mergeModule);
          the -= the % mergeTheta[l];
          ev.cells[l | ((mod + 2048) << 4) | (the << 16)] += e;
        }
      }
    }
    // anything leaving the module into the world is counted and killed; the back face is the virtual detector
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
  if (argc < 3) { G4cerr << "usage: sim run.mac out.root [seed=1] [backAl_mm=0]" << G4endl; return 1; }
  G4Random::setTheSeed(argc > 3 ? std::atol(argv[3]) : 1);
  layR[0] = Rmin;
  double acc = 0;
  for (int l = 0; l < nLay; ++l) { acc += layLen[l] * cm; layR[l + 1] = rOfS(acc); }
  // ponytail: sequential run manager, event data in file-scope globals; parallelise with one job per seed
  auto rm = new G4RunManager;
  const char* pl = std::getenv("PHYSLIST");
  rm->SetUserInitialization(G4PhysListFactory().GetReferencePhysList(pl ? pl : "QGSP_BERT"));
  rm->SetUserInitialization(new Det(argc > 4 ? std::atof(argv[4]) * mm : 0));
  rm->SetUserAction(new Gun);
  rm->SetUserAction(new Run(argv[2]));
  rm->SetUserAction(new Evt);
  rm->SetUserAction(new Step);
  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  delete rm;
}
