# copyright ############################### #
# This file is part of the Xcoll Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import os
import requests
from subprocess import run

from ...package_env import BaseInterface
from ...general import _pkg_root
try:
    from xaux import FsPath  # TODO: once xaux is in Xsuite keep only this
except (ImportError, ModuleNotFoundError):
    from ...xaux import FsPath


class BdsimInterface(BaseInterface):
    _read_only_paths = {'bdsim': 0, 'geant4': 0}

    def __init__(self):
        super().__init__()
        self._in_constructor = True
        self._geant4 = None
        self._bdsim = None
        self._in_constructor = False
        self._geant4_sourced = False
        self._bdsim_sourced = False
        try:
            cmd = run(['which', 'geant4-config'], capture_output=True)
        except FileNotFoundError:
            pass
        else:
            if cmd.returncode == 0:
                path = FsPath(cmd.stdout.decode().strip())
                if path.exists():
                    self._geant4 = path
                    self._geant4_sourced = True
        try:
            cmd = run(['which', 'bdsim'], capture_output=True)
        except FileNotFoundError:
            pass
        else:
            if cmd.returncode == 0:
                path = FsPath(cmd.stdout.decode().strip())
                if path.exists():
                    self._bdsim = path
                    self._bdsim_sourced = True

    @property
    def compiled(self):
        if self.geant4 is None or self.bdsim is None:
            return False
        try:
            import bdsim
            return hasattr(bdsim, 'BDSLinkTrackerInterface')
        except (ModuleNotFoundError, ImportError) as error:
            return False

    @property
    def ready(self):
        return super().ready and self._geant4_sourced and self._bdsim_sourced

    def compile(self, verbose=True, verbose_compile_output=False):
        # Check all dependencies
        # not needed
        pass

    def assert_geant4_installed(self):
        if self.geant4 is None:
            raise RuntimeError("Could not find Geant4 installation! Please install Geant4.")
        if not self._geant4_sourced:
            raise RuntimeError(f"Geant4 installation found in {self.geant4} "
                               f"but not active! Please source environment.")

    def assert_bdsim_installed(self):
        if self.bdsim is None:
            raise RuntimeError("Could not find BDSIM installation! Please install BDSIM.")
        if not self._bdsim_sourced:
            raise RuntimeError(f"BDSIM installation found in {self.bdsim} "
                               f"but not active! Please source environment.")

    def assert_environment_ready(self):
        self.assert_geant4_installed()
        self.assert_bdsim_installed()
        super().assert_environment_ready()


    def get_bdsim_version(self):
        cmd = run(['bdsim', '--version'], capture_output=True)
        if cmd.returncode != 0:
            stderr = cmd.stderr.decode('UTF-8').strip()
            raise RuntimeError(f"Failed to run 'bdsim --version'!\nError given is:\n{stderr}")
        return cmd.stdout.decode('UTF-8').strip()

    def bdsim_older_than(self, bdsim_version=None, compare_version='1.7.7.develop'):
        if bdsim_version is None:
            bdsim_version = self.get_bdsim_version()
        n_ver = sum([10**(3*(2-i))*int(j) for i, j in enumerate(bdsim_version.strip().split('.')[:3])])
        if 'develop' in bdsim_version:
            n_ver += 0.5
        n_ver_cmp = sum([10**(3*(2-i))*int(j) for i, j in enumerate(compare_version.strip().split('.')[:3])])
        if 'develop' in compare_version:
            n_ver_cmp += 0.5
        return n_ver < n_ver_cmp
