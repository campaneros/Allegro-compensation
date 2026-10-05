// Reads what the simulation needs directly from the k4geo compact files of the ALLEGRO ECAL barrel
// (DectDimensions.xml, ECalBarrel_thetamodulemerged.xml, elements.xml, materials.xml).
// Expressions are evaluated with the CLHEP evaluator, the same one DD4hep is built on. Lengths in mm, angles in rad.
#pragma once
#include <CLHEP/Evaluator/Evaluator.h>
#include <cmath>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct AllegroXml {
  struct Mat { double density; bool fractions; std::vector<std::pair<std::string, double>> comp; };   // g/cm3
  double rmin, rmax, dz, cryoRmin1, cryoRmin2, cryoRmax1, cryoRmax2, angle;
  double tActive, tReadout, tInner, tGlue, tOuter, thetaGrid, thetaOffset;
  int nPlanes;
  std::vector<double> layers;                 // lengths along the electrode
  std::vector<int> mergeTheta, mergeModule;   // per layer
  std::map<std::string, std::string> role;    // active, core0, core, glue, outer, readout, cryo -> material name
  std::map<std::string, Mat> mat;

  explicit AllegroXml(const std::string& dir) {
    std::string dims = slurp(dir + "/DectDimensions.xml"), ecal = slurp(dir + "/ECalBarrel_thetamodulemerged.xml");
    std::string mats = slurp(dir + "/elements.xml") + slurp(dir + "/materials.xml");

    // constants, in Geant4 units (mm = 1); several passes because a constant may use one defined further down
    ev_.setStdMath();
    ev_.setSystemOfUnits(1.e+3, 1. / 1.60217733e-25, 1.e+9, 1. / 1.60217733e-10, 1.0, 1.0, 1.0);
    std::vector<std::pair<std::string, std::string>> todo;
    std::regex reC("<constant\\s[^>]*>");
    for (const std::string& f : {dims, ecal})
      for (std::sregex_iterator it(f.begin(), f.end(), reC), end; it != end; ++it) todo.push_back({attr(it->str(), "name"), attr(it->str(), "value")});
    for (bool progress = true; progress;) {
      progress = false;
      for (auto& [name, expr] : todo) {
        if (name.empty()) continue;
        double v = ev_.evaluate(expr.c_str());
        if (ev_.status() != HepTool::Evaluator::OK) continue;
        ev_.setVariable(name.c_str(), v);
        name.clear();
        progress = true;
      }
    }

    // the barrel detector element
    std::string det = block(ecal, "detector");
    if (attr(det, "type").find("ECalBarrel_NobleLiquid_InclinedTrapezoids") == std::string::npos) throw std::runtime_error("unexpected ECAL barrel detector type " + attr(det, "type"));
    std::string cryo = block(det, "cryostat"), calo = block(det, "calorimeter");
    std::string passive = block(calo, "passive"), active = block(calo, "active"), readout = block(calo, "readout");
    std::string cd = block(calo, "dimensions"), kd = block(cryo, "dimensions");
    rmin = num(cd, "rmin"); rmax = num(cd, "rmax"); dz = num(cd, "dz");
    cryoRmin1 = num(kd, "rmin1"); cryoRmin2 = num(kd, "rmin2"); cryoRmax1 = num(kd, "rmax1"); cryoRmax2 = num(kd, "rmax2");
    angle = num(block(passive, "rotation"), "angle");
    tActive = num(active, "thickness"); tReadout = num(readout, "thickness");
    tInner = num(block(passive, "inner"), "thickness"); tGlue = num(block(passive, "glue"), "thickness"); tOuter = num(block(passive, "outer"), "thickness");
    if (std::abs(num(block(passive, "innerMax"), "thickness") - tInner) > 1e-9) throw std::runtime_error("trapezoidal absorbers (innerMax != inner thickness) are not implemented");
    // number of planes: same formula as ECalBarrel_NobleLiquid_InclinedTrapezoids_o1_v03_geo.cpp
    nPlanes = std::lround(M_PI / std::asin((tInner + tGlue + tOuter + tActive + tReadout) / (2. * rmin * std::cos(angle))));
    std::string lays = block(calo, "layers");
    std::regex reL("<layer\\s[^>]*>");
    for (std::sregex_iterator it(lays.begin(), lays.end(), reL), end; it != end; ++it) {
      std::string rep = attr(it->str(), "repeat");
      for (int i = 0, n = rep.empty() ? 1 : std::stoi(rep); i < n; ++i) layers.push_back(num(it->str(), "thickness"));
    }

    // readout segmentation of that detector
    std::smatch m;
    if (!std::regex_search(ecal, m, std::regex("<readout\\s[^>]*name=\"" + attr(det, "readout") + "\"[^>]*>"))) throw std::runtime_error("readout of the ECAL barrel not found");
    std::string seg = block(ecal.substr(m.position()), "segmentation");
    thetaGrid = num(seg, "grid_size_theta"); thetaOffset = num(seg, "offset_theta");
    if (std::lround(num(seg, "nModules")) != nPlanes) throw std::runtime_error("nModules of the segmentation != number of planes from the geometry");
    std::istringstream a(attr(seg, "mergedCells_Theta")), b(attr(seg, "mergedModules"));
    for (int v; a >> v;) mergeTheta.push_back(v);
    for (int v; b >> v;) mergeModule.push_back(v);
    if (mergeTheta.size() != layers.size() || mergeModule.size() != layers.size()) throw std::runtime_error("segmentation merging lists do not match the number of layers");

    // materials
    role = {{"active", matName(active)}, {"core0", matName(block(passive, "inner"))}, {"core", matName(block(passive, "innerMax"))},
            {"glue", matName(block(passive, "glue"))}, {"outer", matName(block(passive, "outer"))},
            {"readout", matName(readout)}, {"cryo", matName(cryo)}};
    for (auto& [r, name] : role) {
      if (mat.count(name)) continue;
      if (!std::regex_search(mats, m, std::regex("<material\\s[^>]*name=\"" + name + "\"[^>]*>"))) throw std::runtime_error("material " + name + " not found");
      std::string mb = mats.substr(m.position(), mats.find("</material>", m.position()) - m.position());
      std::string D = block(mb, "D");
      if (!attr(D, "unit").empty() && attr(D, "unit") != "g/cm3") throw std::runtime_error("material " + name + ": density unit not g/cm3");
      Mat x{std::stod(attr(D, "value")), false, {}};
      std::regex reK("<(composite|fraction)\\s[^>]*>");
      int nFrac = 0;
      for (std::sregex_iterator it(mb.begin(), mb.end(), reK), end; it != end; ++it) {
        nFrac += (*it)[1] == "fraction";
        x.comp.push_back({attr(it->str(), "ref"), std::stod(attr(it->str(), "n"))});
      }
      if (x.comp.empty() || (nFrac && nFrac != int(x.comp.size()))) throw std::runtime_error("material " + name + ": unsupported composition");
      x.fractions = nFrac > 0;
      mat[name] = x;
    }
  }

private:
  HepTool::Evaluator ev_;
  static std::string slurp(const std::string& fn) {   // file content without xml comments
    std::ifstream f(fn);
    if (!f) throw std::runtime_error("cannot open " + fn);
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    for (size_t a; (a = s.find("<!--")) != std::string::npos;) s.erase(a, s.find("-->", a) + 3 - a);
    return s;
  }
  // first element <tag ...> in s, up to its closing tag (or just the tag if self-closing)
  static std::string block(const std::string& s, const std::string& tag) {
    std::smatch m;
    if (!std::regex_search(s, m, std::regex("<" + tag + "(\\s[^>]*)?>"))) throw std::runtime_error("xml: <" + tag + "> not found");
    if (m.str().size() > 1 && m.str()[m.str().size() - 2] == '/') return m.str();
    size_t end = s.find("</" + tag + ">", m.position());
    if (end == std::string::npos) throw std::runtime_error("xml: </" + tag + "> not found");
    return s.substr(m.position(), end - m.position());
  }
  // attribute of the opening tag of an element ("" if absent)
  static std::string attr(const std::string& elem, const std::string& name) {
    std::string open = elem.substr(0, elem.find('>'));
    std::smatch m;
    return std::regex_search(open, m, std::regex("\\s" + name + "\\s*=\\s*\"([^\"]*)\"")) ? m[1].str() : "";
  }
  double num(const std::string& elem, const std::string& name) {
    std::string e = attr(elem, name);
    double v = ev_.evaluate(e.c_str());
    if (e.empty() || ev_.status() != HepTool::Evaluator::OK) throw std::runtime_error("xml: cannot evaluate " + name + "=\"" + e + "\"");
    return v;
  }
  static std::string matName(const std::string& blk) { return attr(block(blk, "material"), "name"); }
};
