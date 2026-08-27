# copyright ############################### #
# This file is part of the Xcoll Package.   #
# Copyright (c) CERN, 2025                  #
# ######################################### #

import gc
import numpy as np
from numbers import Number

import xobjects as xo

from .rpyc import launch_rpyc_with_port # remove after geant4 bugfix
from .bdsim_config import create_bdsim_config_file, get_collimators_from_input_file
from ..geant4.reference_masses import geant4_masses_src
from ..engine import BaseEngine
from ...general import _pkg_root


class BdsimEngine(BaseEngine):

    _xofields = BaseEngine._xofields | {
        '_relative_energy_cut':        xo.Float64
        # 'random_freeze_state':         xo.Int64,  # to be implemented; number of randoms already sampled, such that this can be taken up again later
    }

    _int32 = True
    _uses_input_file = True
    _uses_run_folder = True

    _depends_on = [BaseEngine]

    _extra_c_sources = [geant4_masses_src]

    def __init__(self, **kwargs):
        # Set element classes dynamically
        from ...beam_elements import BdsimElement
        self.__class__._element_classes = (BdsimElement)
        # Initialise geant4-only defaults
        self._link = None
        self._element_index = None
        self._server = None # remove after geant4 bugfix
        self._conn = None # remove after geant4 bugfix
        super().__init__(**kwargs)
        self._already_started = False
        # Set default values for new properties
        self.relative_energy_cut = None
        self.reentry_protection_enabled = None
        self._physics_settings._use_cuts = False


    # ======================
    # === New Properties ===
    # ======================

    @property
    def relative_energy_cut(self):
        return self._relative_energy_cut

    @relative_energy_cut.setter
    def relative_energy_cut(self, val):
        if val is None:
            val = 0.1
        if not isinstance(val, Number) or val <= 0:
            self.stop()
            raise ValueError("`relative_energy_cut` has to be a strictly postive number!")
        self._relative_energy_cut = val

    @property
    def reentry_protection_enabled(self):
        return self._reentry_protection_enabled

    @reentry_protection_enabled.setter
    def reentry_protection_enabled(self, val):
        if val is False:
            print("Warning: Disabling re-entry protection can lead to crashes!")
            print("         Only disable if you know what you are doing.")
        elif val is None:
            try:
                import rpyc
            except ImportError as e:
                val = False
            else:
                val = True
        elif not isinstance(val, bool):
            self.stop()
            raise ValueError("`reentry_protection_enabled` has to be a boolean!")
        self._reentry_protection_enabled = val

    # ============================
    # === Overwrite Properties ===
    # ============================

    @property
    def capacity(self):
        return None  # Geant4 capacity is dynamic

    @property
    def relative_capacity(self):
        return None  # Geant4 capacity is dynamic

    # =================================
    # === Base methods to overwrite ===
    # =================================

    def _set_engine_properties(self, **kwargs):
        kwargs = super()._set_engine_properties(**kwargs)
        self._set_property('relative_energy_cut', kwargs)
        self._set_property('reentry_protection_enabled', kwargs)
        return kwargs

    def _pre_input(self, **kwargs):
        coll_id = 1
        for el in self._element_dict.values():
            el.geant4_id = f'XcollG4.{coll_id}'  # TODO: will be provided by new BDSIM interface
            coll_id += 1
        return kwargs

    def _generate_input_file(self, **kwargs):
        input_file, kwargs = create_bdsim_config_file(element_dict=self._element_dict,
                                particle_ref=self.particle_ref, verbose=self.verbose,
                                cwd=self.cwd, **kwargs)
        # The only thing left in kwargs are parameters to start the engine
        return input_file, kwargs

    def _start_engine(self, **kwargs):
        from ...beam_elements import BaseCrystal, Geant4CollimatorTip
        import bdsim

        # --- create the BDSIM link interface ---
        # reference kinetic energy, converted eV -> GeV

        Ekin_GeV = (self.particle_ref.energy0[0] - self.particle_ref.mass0) * 1.e-9

        if self._already_started:
            # GetInstance is a singleton; Geant4 can't re-initialize in one process
            self.stop(clean=True)
            raise RuntimeError("Cannot restart BDSIM engine in the same Python process.")

        try:
            self._link = bdsim.BDSLinkTrackerInterface.GetInstance(
                                self.input_file.as_posix(),
                                referenceParticlePDG=int(self.particle_ref.pdg_id[0]),
                                referenceKineticEnergy=Ekin_GeV,
                                relativeEnergyCut=self.relative_energy_cut,
                                batchMode=True)
        except (ModuleNotFoundError, ImportError) as error:
            self.stop(clean=True)
            self._warn(error)
            return

        self._bdsim = self._link.GetBDSIMLink()
        self._element_index = {}   # {name: BDSIM link integer}, reset every run

        self.reentry_protection_enabled = False # TODO
        #if self.reentry_protection_enabled:
        #else:

        # --- PART 2 (CHANGED): install collimators, if any were given ---------------------
        if self._element_dict:
            for el in self._element_dict.values():
                self._install_element(el)

        self._already_started = True

    def install(self, element):
        """Install a collimator into the running BDSIM engine.

        Parameters
        ----------
        element : BdsimElement
            The element to install. It must already be part of the engine's
            element dict (i.e. registered via start(elements=...) or the line).
        """
        import bdsim
        if not self.is_running():
            raise RuntimeError("Cannot install: BDSIM engine is not running. "
                               "Call start() first.")

        # A ready-made BDSIM Element: the user has set everything themselves,
        # so hand it straight to BDSIM without xcoll's geometry translation.
        if isinstance(element, bdsim.Element):
            link_id = self._bdsim.AddLinkElement(element)
            self._element_index[element.name] = link_id
            return

        # An xcoll element: translate and install via the internal helper.
        if isinstance(element, self._element_classes):
            self._install_element(element)
            return

        raise TypeError(f"Cannot install object of type {type(element).__name__}.")

    def _install_element(self, el):
        import bdsim
        if self._bdsim is None:
            raise RuntimeError("Cannot install element: BDSIM engine is not running. Call start() first.")
        if el.geant4_id in self._element_index:
            self._print(f"Element {el.geant4_id} already installed. Skipping...")
            return
            # ---- geometry -----
        side = 2 if el._side == -1 else el._side          # NEW (was dropped)
        jaw_L = 0.1 if el.jaw_L is None else el.jaw_L
        jaw_R = -0.1 if el.jaw_R is None else el.jaw_R
        tilt_L = 0.0 if el.tilt_L is None else el.tilt_L
        tilt_R = 0.0 if el.tilt_R is None else el.tilt_R
        jaw_L -= 1.e-9  # Correct for 1e-9 shift that is added in BDSIM
        jaw_R += 1.e-9  # Correct for 1e-9 shift that is added in BDSIM
        if jaw_L < jaw_R:
            self.stop(clean=True)
            raise ValueError(f"BdsimCollimator {el.name} has jaw_L < jaw_R: "
                         + f"jaw_L={el.jaw_L}, jaw_R={el.jaw_R}!")
        xOffset = 0
        yOffset = 0
        if jaw_L <= 0:
            xOffset = (jaw_L + jaw_R)/2
            jaw_L -= xOffset
            jaw_R -= xOffset
            xOffset_temp = xOffset*np.cos(np.deg2rad(el.angle_L)) - yOffset*np.sin(np.deg2rad(el.angle_L))
            yOffset = xOffset*np.sin(np.deg2rad(el.angle_L)) + yOffset*np.cos(np.deg2rad(el.angle_L))
            xOffset = xOffset_temp
        if jaw_R >= 0:
            xOffset = (jaw_L + jaw_R)/2
            jaw_L -= xOffset
            jaw_R -= xOffset
            xOffset_temp = xOffset*np.cos(np.deg2rad(el.angle_R)) - yOffset*np.sin(np.deg2rad(el.angle_R))
            yOffset = xOffset*np.sin(np.deg2rad(el.angle_R)) + yOffset*np.cos(np.deg2rad(el.angle_R))
            xOffset = xOffset_temp

        # ---- one-sidedness: generic path has no buildLeft/buildRight, so an
        #      unbuilt jaw is expressed by an aperture > horizontalWidth/2 ----
        horizontal_width = 2.0                            # NEW / VERIFY
        build_left  = side in (0, 1)                      # NEW (was dropped)
        build_right = side in (0, 2)                      # NEW (was dropped)
        xsize_left  =  jaw_L if build_left  else horizontal_width
        xsize_right = -jaw_R if build_right else horizontal_width

        # ---- build a generic Element and add it ----
        g4el = bdsim.Element()
        g4el.type = bdsim.elementtype.ElementType.JCOL
        g4el.name = f'{el.geant4_id}'
        g4el.set_value('material',        el.material.geant4_name)
        g4el.set_value('l',               el.length)
        g4el.set_value('xsizeLeft',       xsize_left)
        g4el.set_value('xsizeRight',      xsize_right)
        g4el.set_value('ysize',           0.2)            # NEW — half size, m; matches old builder
        g4el.set_value('tilt',            np.deg2rad(el.angle))
        g4el.set_value('offsetX',         xOffset)
        g4el.set_value('offsetY',         yOffset)
        g4el.set_value('jawTiltLeft',     tilt_L)
        g4el.set_value('jawTiltRight',    tilt_R)
        g4el.set_value('horizontalWidth', horizontal_width)
        link_id = self._bdsim.AddLinkElement(g4el)
        self._element_index[el.geant4_id] = link_id

    def _stop_engine(self, **kwargs):
        del self._link
        gc.collect()
        self._link = None
        self._element_index = {}
        if self.reentry_protection_enabled and self._server: # remove after geant4 bugfix
            self._server.terminate() # remove after geant4 bugfix
            self._server = None # remove after geant4 bugfix
        return kwargs

    def _is_running(self):
        return self._link is not None

    def _get_input_files_to_clean(self, input_file, cwd, **kwargs):
        if cwd is None or input_file is None:
            return []
        return [cwd / input_file]

    def _get_output_files_to_clean(self, input_file, cwd, **kwargs):
        if cwd is None:
            return []
        files_to_delete = ['rpyc.log', 'geant4.out', 'geant4.err',
                           'engine.out', 'engine.err', 'root.out',
                           'root.err']
        return [cwd / f for f in files_to_delete]

    def _match_input_file(self):
        # Read the elements in the input file and compare to the elements in the engine,
        # overwriting parameters where necessary
        input_dict = get_collimators_from_input_file(self.input_file)
        for name in input_dict:
            if name not in self._element_dict:
                self.stop()
                raise ValueError(f"Element {name} in input file not found in engine!")
        for name, ee in self._element_dict.items():
            from ...beam_elements import Geant4CollimatorTip
            if name not in input_dict:
                self._print(f"Warning: Geant4Collimator {name} not in Geant4 input file! "
                          + f"Maybe it was fully open. Deactivated")
                self._deactivate_element(ee)
                continue
            self._assert_element(ee)
            ee.geant4_id = input_dict[name]['geant4_id']
            if not np.isclose(ee.length, input_dict[name]['length'], atol=1e-9):
                self._print(f"Warning: Length of {name} differs from input file "
                        + f"({ee.length} vs {input_dict[name]['length']})! Overwritten.")
                ee.length = input_dict[name]['length']
            if not np.isclose(ee.angle, input_dict[name]['angle'], atol=1e-9):
                self._print(f"Warning: Angle of {name} differs from input file "
                        + f"({ee.angle} vs {input_dict[name]['angle']})! Overwritten.")
                ee.angle = input_dict[name]['angle']
            if ee.material.geant4_name != input_dict[name]['material'] \
            and ee.material.name != input_dict[name]['material']:
                self.stop()
                raise ValueError(f"Material of {name} differs from input file "
                            + f"({ee.material.geant4_name or ee.material.name} "
                            + f"vs {input_dict[name]['material']})!")
            if isinstance(ee, Geant4CollimatorTip) or 'tip_material' in input_dict[name]:
                if not isinstance(ee, Geant4CollimatorTip):
                    self.stop()
                    raise ValueError(f"Element {name} is not a Geant4CollimatorTip "
                                    + "in the line, but it has tip material in the input file!")
                if 'tip_material' not in input_dict[name] or 'tip_thickness' not in input_dict[name]:
                    self.stop()
                    raise ValueError(f"Element {name} is a Geant4CollimatorTip, "
                                    + "but it has no tip material in the input file!")
                if ee.tip_material.geant4_name != input_dict[name]['tip_material']:
                    self._print(f"Warning: Tip material of {name} differs from input file "
                            + f"({ee.tip_material.geant4_name} vs {input_dict[name]['tip_material']})! Overwritten.")
                    ee.tip_material.geant4_name = input_dict[name]['tip_material']
                if not np.isclose(ee.length, input_dict[name]['tip_thickness'], atol=1e-9):
                    self._print(f"Warning: Tip thickness of {name} differs from input file "
                            + f"({ee.tip_thickness} vs {input_dict[name]['tip_thickness']})! Overwritten.")
                    ee.tip_thickness = input_dict[name]['tip_thickness']
            jaw = input_dict[name]['jaw']
            if jaw is not None and not hasattr(jaw, '__iter__'):
                jaw = [jaw, -jaw]
            if jaw is None or (jaw[0] is None and jaw[1] is None):
                ee.jaw = None
            else:
                if jaw[0] is None:
                    if ee.side != 'right':
                        self._print(f"Warning: {name} is right-sided in the input file, "
                                + "but not in the line! Overwritten by the former.")
                        ee.side = 'right'
                elif ee.jaw_L is None or not np.isclose(ee.jaw_L, jaw[0], atol=1e-9):
                    self._print(f"Warning: Jaw_L of {name} differs from input file "
                            + f"({ee.jaw_L} vs {jaw[0]})! Overwritten.")
                    ee.jaw_L = jaw[0]
                if jaw[1] is None:
                    if ee.side != 'left':
                        self._print(f"Warning: {name} is left-sided in the input file, "
                                + f"but not in the line! Overwritten by the former.")
                        ee.side = 'left'
                elif ee.jaw_R is None or not np.isclose(ee.jaw_R, jaw[1], atol=1e-9):
                    self._print(f"Warning: Jaw_R of {name} differs from input file "
                            + f"({ee.jaw_R} vs {jaw[1]})! Overwritten.")
                    ee.jaw_R = jaw[1]
            tilt = input_dict[name]['tilt']
            if not hasattr(tilt, '__iter__'):
                tilt = [tilt, -tilt]
            if ee.side != 'right' and not np.isclose(ee.tilt_L, tilt[0], atol=1e-9):
                self._print(f"Warning: Tilt_L of {name} differs from input file "
                        + f"({ee.tilt_L} vs {tilt[0]})! Overwritten.")
                ee.tilt_L = tilt[0]
            if ee.side != 'left' and not np.isclose(ee.tilt_R, tilt[1], atol=1e-9):
                self._print(f"Warning: Tilt_R of {name} differs from input file "
                        + f"({ee.tilt_R} vs {tilt[1]})! Overwritten.")
                ee.tilt_R = tilt[1]
