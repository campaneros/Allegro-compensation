#!/bin/bash
# Usage (lxplus): ./run_sim.sh <particle> <nEvents> <seed>      e.g. ./run_sim.sh pi- 2000 1
# Produces sim_<particle>_<seed>.root; a virtual detector behind the ECAL cryostat records and kills everything that exits.
set -e
PART=${1:-pi-}; N=${2:-1000}; SEED=${3:-1}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source /cvmfs/sw.hsf.org/key4hep/setup.sh
D=geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03
if [ ! -f steering.py ]; then   # setup not completed yet
  rm -rf geo; cp -r "$K4GEO" geo   # xml from the release, so it matches the installed detector plugins
  cp VirtualDetector.xml $D/
  # ECAL barrel alone (cryostat + solenoid + field included) + virtual detector: nothing upstream of the calorimeter
  grep -v -E 'Beampipe|BeamInstrumentation|LumiCal|Vertex|DriftChamber|SiliconWrapper|HCal|ECalEndcaps|MuonTagger|InstallSurfaceManager' \
    $D/ALLEGRO_o1_v03.xml | sed 's|<include ref="ECalBarrel_thetamodulemerged.xml"/>|&\n  <include ref="VirtualDetector.xml"/>|' > $D/ALLEGRO_ECalOnly.xml
  # own steering: the official one applies the 15 mrad crossing-angle boost to the gun particle
  printf '%s\n' 'from DDSim.DD4hepSimulation import DD4hepSimulation' 'SIM = DD4hepSimulation()' \
    'SIM.action.mapActions["VD"] = "Geant4EscapeCounter"' > steering.py
fi
ddsim --steeringFile steering.py \
  --compactFile $D/ALLEGRO_ECalOnly.xml \
  --enableGun --gun.particle "$PART" --gun.distribution uniform \
  --gun.momentumMin "5*GeV" --gun.momentumMax "100*GeV" \
  --gun.thetaMin "70*deg" --gun.thetaMax "110*deg" --gun.phiMin "0*deg" --gun.phiMax "360*deg" \
  --physics.list QGSP_BERT \
  --numberOfEvents "$N" --random.seed "$SEED" --outputFile "sim_${PART}_${SEED}.root"
