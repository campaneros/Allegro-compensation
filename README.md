# allegro-compensation-standalone

Software compensation of pion showers (arXiv:2606.05111, Figs. 4-12) on the ALLEGRO ECAL barrel used as a test-beam
module, simulated with standalone Geant4 (no key4hep).

## Geometry
The full ECAL barrel of ALLEGRO, read from the k4geo compact files and built as the k4geo driver
`ECalBarrel_NobleLiquid_InclinedTrapezoids_o1_v03_geo.cpp` does:

    python xml2geo.py /path/to/k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03 > geometry.txt

`xml2geo.py` evaluates the constants of `DectDimensions.xml` and `ECalBarrel_thetamodulemerged.xml` and writes
dimensions, plate structure, layers, readout segmentation and materials to `geometry.txt`, which both `sim` and
`ntuple.py` read. The committed `geometry.txt` comes from k4geo commit e1ba7bc.

- Complete ring (all 1536 planes) and full length along z: no lateral leakage in the calorimeter itself.
- Front cryostat wall, liquid bath, back cryostat wall (which includes the solenoid thickness). Set
  `use_cryo_back 0` in `geometry.txt` to remove the back wall and put the virtual detector right behind the liquid.
- Not modelled: cryostat side walls at the z ends, magnetic field, everything outside the ECAL barrel.
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

4th argument of `sim`: geometry file (default `geometry.txt` in the current directory).
`PHYSLIST=FTFP_BERT ./build/sim ...` changes the physics list (default QGSP_BERT).

## Definitions
- `Ebeam`: total energy of the generated particle. `Evd`: energy leaving through the back face
  (kinetic for baryons, total otherwise). `Etrue = Ebeam - Evd`. `Eside`: energy leaving backwards into the bore or through the z ends.
- `Rend`: radius where the primary ended (first inelastic interaction, or exit).
- hits: LAr deposits per cell, calibrated per layer to the EM scale, threshold 3 MeV (`--thr`).
