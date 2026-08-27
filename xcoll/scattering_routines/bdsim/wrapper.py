# copyright ############################### #
# This file is part of the Xcoll Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..wrapper import BaseWrapper
from .engine import BdsimEngine
from .environment import BdsimInterface
from ..geant4.reference_masses import geant4_masses_meta


class BdsimWrapper(BaseWrapper):
    """Wrapper for all Geant4 and BDSIM functions."""

    _engine_cls = BdsimEngine
    _interface_cls = BdsimInterface
    _particle_masses_meta = geant4_masses_meta
