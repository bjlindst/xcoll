# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from .general import _pkg_root, __version__, citation

from .beam_elements import BlackAbsorber, BlackCrystal, EverestBlock, EverestCollimator, EverestCrystal, \
                           Geant4Collimator, collimator_classes, crystal_classes, element_classes, \
                           BlowUp, EmittanceMonitor
from .install import install_elements
from .line_tools import assign_optics_to_collimators, open_collimators, send_to_parking, enable_scattering, disable_scattering
from .scattering_routines.everest import materials, Material, CrystalMaterial
from .scattering_routines.geant4 import Geant4Engine
from .colldb import CollimatorDatabase
from .interaction_record import InteractionRecord
from .rf_sweep import RFSweep
from .initial_distribution import *
from .lossmap import LossMap

# Deprecated
from ._manager import CollimatorManager

# print("If you use Xcoll in your simulations, please cite us :-)")
# print(citation)

