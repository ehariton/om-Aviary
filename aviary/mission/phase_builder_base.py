'''
Define utilities for building phases.

Classes
-------
PhaseBuilderBase : the interface for a phase builder
'''
from abc import ABC
from collections import namedtuple

from aviary.mission.initial_guess_builders import InitialGuess
from aviary.variable_info.variables import Dynamic

import dymos as dm
import openmdao.api as om

from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.utils.aviary_values import AviaryValues, get_keys
from aviary.variable_info.variable_meta_data import _MetaData

_require_new_meta_data_class_attr_ = \
    namedtuple('_require_new_meta_data_class_attr_', ())


_require_new_initial_guesses_meta_data_class_attr_ = \
    namedtuple('_require_new_initial_guesses_meta_data_class_attr_', ())


class PhaseBuilderBase(ABC):
    '''
    Define the interface for a phase builder.

    Attributes
    ----------
    name : str ('<unknown phase>')
        object label

    core_subsystems : (None)
        list of SubsystemBuilderBase objects that will be added to the phase ODE

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

    ode_class : type (None)
        advanced: the type of system defining the ODE

    transcription : "Dymos transcription object" (None)
        advanced: an object providing the transcription technique of the
        optimal control problem

    subsystem_options : dict (None)
        dictionary of parameters to be passed to the subsystem builders

    default_name : str
        class attribute: derived type customization point; the default value
        for name

    default_ode_class : type
        class attribute: derived type customization point; the default value
        for ode_class used by build_phase

    is_analytic_phase : bool (False)
        class attribute: derived type customization point; if True, build_phase
        will return an AnalyticPhase instead of a Phase

    num_nodes : int (5)
        class attribute: derived type customization point; the default value
        for num_nodes used by build_phase, only for AnalyticPhases

    Methods
    -------
    build_phase
    make_default_transcription
    validate_options
    assign_default_options
    '''
    __slots__ = (
        'name',  'core_subsystems', 'subsystem_options', 'user_options',
        'initial_guesses', 'ode_class', 'transcription',
        'is_analytic_phase', 'num_nodes', 'external_subsystems', 'meta_data',
    )

    # region : derived type customization points
    _meta_data_ = _require_new_meta_data_class_attr_()

    _initial_guesses_meta_data_ = _require_new_initial_guesses_meta_data_class_attr_()

    default_name = '<unknown phase>'

    default_ode_class = MissionODE

    default_meta_data = _MetaData
    # endregion : derived type customization points

    def __init__(
        self, name=None, core_subsystems=None, user_options=None, initial_guesses=None,
        ode_class=None, transcription=None, subsystem_options=None, is_analytic_phase=False, num_nodes=5, external_subsystems=None, meta_data=None,
    ):
        if name is None:
            name = self.default_name

        self.name = name

        if core_subsystems is None:
            core_subsystems = []

        self.core_subsystems = core_subsystems

        if subsystem_options is None:
            subsystem_options = {}

        self.subsystem_options = subsystem_options

        self.user_options = user_options
        self.validate_options()
        self.assign_default_options()

        if initial_guesses is None:
            initial_guesses = AviaryValues()

        self.initial_guesses = initial_guesses
        self.validate_initial_guesses()

        self.ode_class = ode_class
        self.transcription = transcription
        self.is_analytic_phase = is_analytic_phase
        self.num_nodes = num_nodes

        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data

    def build_phase(self, aviary_options=None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (empty)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        ode_class = self.ode_class

        if ode_class is None:
            ode_class = self.default_ode_class

        transcription = self.transcription

        if transcription is None and not self.is_analytic_phase:
            transcription = self.make_default_transcription()

        if aviary_options is None:
            aviary_options = AviaryValues()

        kwargs = self._extra_ode_init_kwargs()

        kwargs = {
            'aviary_options': aviary_options,
            **kwargs
        }

        subsystem_options = self.subsystem_options

        if subsystem_options is not None:
            kwargs['subsystem_options'] = subsystem_options

        kwargs['core_subsystems'] = self.core_subsystems

        if self.is_analytic_phase:
            phase = dm.AnalyticPhase(
                ode_class=ode_class,
                ode_init_kwargs=kwargs,
                num_nodes=self.num_nodes,
            )
        else:
            phase = dm.Phase(
                ode_class=ode_class, transcription=transcription,
                ode_init_kwargs=kwargs
            )

        # overrides should add state, controls, etc.
        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        user_options = self.user_options

        num_segments, _ = user_options.get_item('num_segments')
        order, _ = user_options.get_item('order')

        transcription = dm.Radau(
            num_segments=num_segments, order=order, compressed=True)

        return transcription

    def validate_options(self):
        '''
        Raise TypeError if an unsupported option is found.

        Users can call this method when updating options after initialization.
        '''
        user_options = self.user_options

        if not user_options:
            return  # acceptable

        meta_data = self._meta_data_

        for key in get_keys(user_options):
            if key not in meta_data:
                raise TypeError(
                    f'{self.__class__.__name__}: {self.name}:'
                    f' unsupported option: {key}'
                )

    def assign_default_options(self):
        '''
        Update missing options with default values.

        If user_options is None, start with an empty AviaryValues.

        Users can call this method when replacing the user_options member after
        initialization.
        '''
        user_options = self.user_options

        if user_options is None:
            user_options = self.user_options = AviaryValues()

        meta_data = self._meta_data_

        for key in meta_data:
            if key not in user_options:
                item = meta_data[key]

                val = item['val']
                units = item['units']

                user_options.set_val(key, val, units)

    def validate_initial_guesses(self):
        '''
        Raise TypeError if an unsupported initial guess is found.

        Users can call this method when updating initial guesses after initialization.
        '''
        initial_guesses = self.initial_guesses

        if not initial_guesses:
            return  # acceptable

        meta_data = self._initial_guesses_meta_data_

        for key in get_keys(initial_guesses):
            if key not in meta_data:
                raise TypeError(
                    f'{self.__class__.__name__}: {self.name}:'
                    f' unsupported initial guess: {key}'
                )

    def apply_initial_guesses(
        self, prob: om.Problem, traj_name, phase: dm.Phase
    ):
        '''
        Apply any stored initial guesses; return a list of guesses not applied.
        '''
        not_applied = []

        phase_name = self.name
        meta_data = self._initial_guesses_meta_data_
        initial_guesses: AviaryValues = self.initial_guesses

        for key in meta_data:
            if key in initial_guesses:
                apply_initial_guess = meta_data[key]['apply_initial_guess']
                val, units = initial_guesses.get_item(key)

                apply_initial_guess(prob, traj_name, phase, phase_name, val, units)

            else:
                not_applied.append(key)

        return not_applied

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {}

    def to_phase_info(self):
        '''
        Return the stored settings as phase info.

        Returns
        -------
        tuple
            name : str
                object label
            phase_info : dict
                stored settings
        '''
        subsystem_options = self.subsystem_options  # TODO: aero info?
        user_options = dict(self.user_options)
        initial_guesses = dict(self.initial_guesses)

        # TODO some of these may be purely programming API hooks, rather than for use
        # with phase info
        # - ode_class
        # - transcription
        # - external_subsystems
        # - meta_data

        phase_info = dict(
            subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses)

        return (self.name, phase_info)

    @classmethod
    def from_phase_info(cls, name, phase_info: dict, core_subsystems=None, meta_data=None, transcription=None):
        '''
        Return a new phase builder based on the specified phase info.

        Note, calling code is responsible for matching phase info to the correct phase
        builder type, or the behavior is undefined.

        Parameters
        ----------
        name : str
            object label
        phase_info : dict
            stored settings
        '''
        # loop over user_options dict entries
        # if the value is not a tuple, wrap it in a tuple with the second entry of 'unitless'
        for key, value in phase_info['user_options'].items():
            if not isinstance(value, tuple):
                phase_info['user_options'][key] = (value, 'unitless')

        subsystem_options = phase_info.get(
            'subsystem_options', {})  # TODO: aero info?
        user_options = AviaryValues(phase_info.get('user_options', ()))
        initial_guesses = AviaryValues(phase_info.get('initial_guesses', ()))
        external_subsystems = phase_info.get('external_subsystems', [])
        # TODO core subsystems in phase info?

        # TODO some of these may be purely programming API hooks, rather than for use
        # with phase info
        # - ode_class
        # - transcription
        # - external_subsystems
        # - meta_data

        phase_builder = cls(
            name, subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses, meta_data=meta_data,
            core_subsystems=core_subsystems, external_subsystems=external_subsystems, transcription=transcription)

        return phase_builder

    @classmethod
    def _add_meta_data(cls, name, *, val, units='unitless', desc=None):
        '''
        Update supported options with a new item.

        Raises
        ------
        ValueError
            if a repeat option is found
        '''
        meta_data = cls._meta_data_

        if name in meta_data:
            raise ValueError(
                f'{cls.__name__}": meta data: repeat option: {name}'
            )

        meta_data[name] = dict(val=val, units=units, desc=desc)

    @classmethod
    def _add_initial_guess_meta_data(cls, initial_guess: InitialGuess, desc=None):
        '''
        Update supported initial guesses with a new item.

        Raises
        ------
        ValueError
            if a repeat initial guess is found
        '''
        meta_data = cls._initial_guesses_meta_data_
        name = initial_guess.key

        meta_data[name] = dict(
            apply_initial_guess=initial_guess.apply_initial_guess, desc=desc)

    def _add_user_defined_constraints(self, phase, constraints):
        # Add each constraint and its corresponding arguments to the phase
        for constraint_name, kwargs in constraints.items():
            if kwargs['type'] == 'boundary':
                kwargs.pop('type')
                phase.add_boundary_constraint(constraint_name, **kwargs)
            elif kwargs['type'] == 'path':
                kwargs.pop('type')
                phase.add_path_constraint(constraint_name, **kwargs)

    def set_time_options(self, user_options, targets=[]):
        fix_initial = user_options.get_val('fix_initial')
        duration_bounds = user_options.get_val('duration_bounds', units='s')
        duration_ref = user_options.get_val('duration_ref', units='s')

        self.phase.set_time_options(
            fix_initial=fix_initial,
            duration_bounds=duration_bounds,
            units="s",
            targets=targets,
            duration_ref=duration_ref,
        )

    def add_velocity_state(self, user_options):
        velocity_lower = user_options.get_val('velocity_lower', units='kn')
        velocity_upper = user_options.get_val('velocity_upper', units='kn')
        velocity_ref = user_options.get_val('velocity_ref', units='kn')
        velocity_ref0 = user_options.get_val('velocity_ref0', units='kn')
        velocity_defect_ref = user_options.get_val('velocity_defect_ref', units='kn')
        self.phase.add_state(
            Dynamic.Mission.VELOCITY,
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=velocity_lower,
            upper=velocity_upper,
            units="kn",
            rate_source=Dynamic.Mission.VELOCITY_RATE,
            targets=Dynamic.Mission.VELOCITY,
            ref=velocity_ref,
            ref0=velocity_ref0,
            defect_ref=velocity_defect_ref,
        )

    def add_mass_state(self, user_options):
        mass_lower = user_options.get_val('mass_lower', units='lbm')
        mass_upper = user_options.get_val('mass_upper', units='lbm')
        mass_ref = user_options.get_val('mass_ref', units='lbm')
        mass_ref0 = user_options.get_val('mass_ref0', units='lbm')
        mass_defect_ref = user_options.get_val('mass_defect_ref', units='lbm')
        self.phase.add_state(
            Dynamic.Mission.MASS,
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=mass_lower,
            upper=mass_upper,
            units="lbm",
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Mission.MASS,
            ref=mass_ref,
            ref0=mass_ref0,
            defect_ref=mass_defect_ref,
        )

    def add_distance_state(self, user_options, units='NM'):
        distance_lower = user_options.get_val('distance_lower', units=units)
        distance_upper = user_options.get_val('distance_upper', units=units)
        distance_ref = user_options.get_val('distance_ref', units=units)
        distance_ref0 = user_options.get_val('distance_ref0', units=units)
        distance_defect_ref = user_options.get_val('distance_defect_ref', units=units)
        self.phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=distance_lower,
            upper=distance_upper,
            units=units,
            rate_source=Dynamic.Mission.DISTANCE_RATE,
            ref=distance_ref,
            ref0=distance_ref0,
            defect_ref=distance_defect_ref,
        )

    def add_flight_path_angle_state(self, user_options):
        angle_lower = user_options.get_val('angle_lower', units='rad')
        angle_upper = user_options.get_val('angle_upper', units='rad')
        angle_ref = user_options.get_val('angle_ref', units='rad')
        angle_ref0 = user_options.get_val('angle_ref0', units='rad')
        angle_defect_ref = user_options.get_val('angle_defect_ref', units='rad')
        self.phase.add_state(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            fix_initial=True,
            fix_final=False,
            lower=angle_lower,
            upper=angle_upper,
            units="rad",
            rate_source=Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            ref=angle_ref,
            defect_ref=angle_defect_ref,
            ref0=angle_ref0,
        )

    def add_altitude_state(self, user_options, units='ft'):
        alt_lower = user_options.get_val('alt_lower', units=units)
        alt_upper = user_options.get_val('alt_upper', units=units)
        alt_ref = user_options.get_val('alt_ref', units=units)
        alt_ref0 = user_options.get_val('alt_ref0', units=units)
        alt_defect_ref = user_options.get_val('alt_defect_ref', units=units)
        self.phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_final=False,
            lower=alt_lower,
            upper=alt_upper,
            units=units,
            rate_source=Dynamic.Mission.ALTITUDE_RATE,
            ref=alt_ref,
            defect_ref=alt_defect_ref,
            ref0=alt_ref0,
        )

    def add_altitude_constraint(self, user_options):
        final_altitude = user_options.get_val('final_altitude', units='ft')
        alt_constraint_ref = user_options.get_val('alt_constraint_ref', units='ft')
        self.phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc="final",
            equals=final_altitude,
            units="ft",
            ref=alt_constraint_ref,
        )


_registered_phase_builder_types = []


def register(phase_builder_t=None, *, check_repeats=True):
    '''
    Register a new phase builder type.

    Note, this function qualifies as a class decorator for ease of use.

    Returns
    -------
    '''
    if phase_builder_t is None:
        def decorator(phase_builder_t):
            return register(phase_builder_t, check_repeats=check_repeats)

        return decorator

    if check_repeats and (phase_builder_t in _registered_phase_builder_types):
        raise ValueError('repeated phase builder type')

    _registered_phase_builder_types.append(phase_builder_t)

    return phase_builder_t


def phase_info_to_builder(name: str, phase_info: dict) -> PhaseBuilderBase:
    '''
    Return a new phase builder based on the specified phase info.

    Note, the type of phase builder will be determined by calling
    phase_builder_t.from_phase_info() for each registered type in order of registration;
    the first result that is not None will be returned. If a supported phase builder type
    cannot be determined, raise ValueError.

    Raises
    ------
    ValueError
        if a supported phase builder type cannot be determined
    '''
    phase_builder_t: PhaseBuilderBase = None

    for phase_builder_t in _registered_phase_builder_types:
        builder = phase_builder_t.from_phase_info(name, phase_info)

        if builder is not None:
            return builder

    raise ValueError(f'unsupported phase info: {name}')


if __name__ == '__main__':
    help(PhaseBuilderBase)

    try:
        PhaseBuilderBase._add_meta_data('test', val=0, units=None, desc='test')

    except Exception as error:
        print(error)
