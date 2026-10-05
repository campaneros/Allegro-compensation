# allegro-compensation-standalone

Software compensation of pion showers (arXiv:2606.05111, Figs. 4-12) on a module of the ALLEGRO ECAL barrel in a
test-beam setup, simulated with standalone Geant4 (no key4hep).

## Geometry
A module of the ALLEGRO ECAL barrel. `sim` reads the k4geo compact files directly at start-up
(`DectDimensions.xml`, `ECalBarrel_thetamodulemerged.xml`, `elements.xml`, `materials.xml`; parser in
`allegro_xml.hh`, expressions evaluated with the CLHEP evaluator) and builds the volumes as the k4geo driver
`ECalBarrel_NobleLiquid_InclinedTrapezoids_o1_v03_geo.cpp` does. No calorimeter number is written in this repository.

    git clone https://github.com/key4hep/k4geo        # in this directory, or pass the xml directory as 4th argument
    ./build/test_xml k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03    # prints what was read from the xml

- Module = a block of consecutive plates, like a supermodule: its sides follow the inclined plates, front and back
  are at constant radius, every plate inside it is complete. Default: plates over +-20 deg in azimuth (171 electrodes,
  about +-0.8 m around the beam at mid depth) and +-1.5 m along z. `/module/halfPhiDeg`, `/module/halfZ` in the macro
  change it; `Eside` (energy leaving through the sides, the z ends or backwards) tells whether it is large enough.
- Front cryostat wall and liquid bath; the virtual detector is right behind the liquid, as in the paper.
  `/module/cryoBack true` adds the back cryostat wall (which includes the solenoid thickness) in front of it.
- No magnetic field, nothing outside the ECAL barrel.
- Cells: gap between two absorbers = module, slices along the electrode = layers, projective theta bins, merged as in the xml.
- Beam (`run.mac`): fired radially from inside the bore, 5 cm in front of the cryostat wall, spread over one readout cell.
  Anything leaving the calorimeter is counted and killed; the outer surface is the virtual detector.

## Run
    cmake -B build && cmake --build build
    ./build/sim run.mac pi_raw_1.root 1            # seed 1; edit particle / energies / events in run.mac
    python ntuple.py make train.root pi_raw_[1-9].root
    python ntuple.py make test.root  pi_raw_10.root
    python gnn.py train.root test.root pred.npy
    python compensation.py train.root test.root --gnn pred.npy -o plots

Calibration: by default the per-layer sampling fractions of the full ALLEGRO simulation (FCC-config). To check
that this module reproduces them: `./build/sim run_e.mac e_raw.root 1 && python ntuple.py calib calib.json e_raw.root`
(prints both sets; `--calib calib.json` in `make` uses the module's own).

4th argument of `sim`: directory of the compact files (default `k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03`).
Each raw file carries the geometry used in a `geo` tree, which `ntuple.py` reads to place the cells.
`PHYSLIST=FTFP_BERT ./build/sim ...` changes the physics list (default QGSP_BERT).

## Looking at the detector
`./build/sim -i` opens an interactive Geant4 session with visualisation (needs a Geant4 built with a graphics
driver, e.g. Qt) and runs `vis.mac`: a thin slice of the module and one 10 GeV pion.

## Definitions
- `Ebeam`: total energy of the generated particle. `Evd`: energy leaving through the back face
  (kinetic for baryons, total otherwise). `Etrue = Ebeam - Evd`. `Eside`: energy leaving through the sides, the z ends or backwards.
- `Rend`: radius where the primary ended (first inelastic interaction, or exit).
- hits: LAr deposits per cell, calibrated per layer to the EM scale, threshold 3 MeV (`--thr`).
