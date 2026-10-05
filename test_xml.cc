// prints what allegro_xml.hh extracts:  ./test_xml /path/to/k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03
#include "allegro_xml.hh"
#include <cstdio>
int main(int argc, char** argv) {
  AllegroXml x(argc > 1 ? argv[1] : "k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03");
  std::printf("rmin %.10g rmax %.10g dz %.10g\ncryo %.10g %.10g %.10g %.10g\nangle %.16g nplanes %d\n", x.rmin, x.rmax, x.dz, x.cryoRmin1, x.cryoRmin2, x.cryoRmax1, x.cryoRmax2, x.angle, x.nPlanes);
  std::printf("t_active %.10g t_readout %.10g t_inner %.10g t_glue %.10g t_outer %.10g\ntheta_grid %.12g theta_offset %.10g\nlayers", x.tActive, x.tReadout, x.tInner, x.tGlue, x.tOuter, x.thetaGrid, x.thetaOffset);
  for (double l : x.layers) std::printf(" %.10g", l);
  std::printf("\nmerge_theta"); for (int v : x.mergeTheta) std::printf(" %d", v);
  std::printf("\nmerge_module"); for (int v : x.mergeModule) std::printf(" %d", v);
  std::printf("\n");
  for (auto& [r, n] : x.role) std::printf("mat_%s %s\n", r.c_str(), n.c_str());
  for (auto& [n, m] : x.mat) { std::printf("material %s %g %s", n.c_str(), m.density, m.fractions ? "fraction" : "natoms"); for (auto& [e, a] : m.comp) std::printf(" %s %g", e.c_str(), a); std::printf("\n"); }
}
