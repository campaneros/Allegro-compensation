# allegro-compensation-standalone

Software compensation of pion showers (arXiv:2606.05111, Figs. 4-12) on a test-beam-like module of the
ALLEGRO ECAL barrel, simulated with standalone Geant4 (no key4hep).

## Module
A phi-sector (+-22 deg, +-0.5 m along z) of the barrel, numbers from k4geo `ALLEGRO_o1_v03`:
1536 planes/2pi inclined by 50.18 deg, absorber 1.8 mm Pb + 0.1 glue + 0.1 steel (LAr core in layer 0),
1.2 mm PCB electrodes, LAr gaps, R = 2172.8-2578.3 mm, 11 layers, cells = 2 modules x 4 theta bins
(1 in layer 1). In front: 13.8 mm Al cryostat wall + 10 mm LAr; behind: 40 mm LAr (+ optional Al).
No magnetic field. Anything leaving the module is counted and killed; the back face is the virtual detector.

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

4th argument of `sim`: mm of aluminium between the bath and the virtual detector (102.7 = back cryostat + solenoid).
`PHYSLIST=FTFP_BERT ./build/sim ...` changes the physics list (default QGSP_BERT).

## Definitions
- `Ebeam`: total energy of the generated particle. `Evd`: energy leaving through the back face
  (kinetic for baryons, total otherwise). `Etrue = Ebeam - Evd`. `Eside`: lateral + backward leakage.
- `Rend`: radius where the primary ended (first inelastic interaction, or exit).
- hits: LAr deposits per cell, calibrated per layer to the EM scale, threshold 3 MeV (`--thr`).
