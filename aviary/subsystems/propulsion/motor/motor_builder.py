from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.motor.model.motor_premission import MotorPreMission
from aviary.subsystems.propulsion.motor.model.motor_mission import MotorMission
from aviary.subsystems.propulsion.motor.motor_variables import Aircraft, Mission, Dynamic
from aviary.subsystems.propulsion.motor.motor_variable_meta_data import ExtendedMetaData


class MotorBuilder(SubsystemBuilderBase):
    '''
    Define the builder for a motor subsystem that provides methods to define the motor subsystem's states, design variables, fixed values, initial guesses, and mass names.

    It also provides methods to build OpenMDAO systems for the pre-mission and mission computations of the subsystem, to get the constraints for the subsystem, and to preprocess the inputs for the subsystem.


    Attributes
    ----------
    name : str ('motor')
        object label

    Methods
    -------
    __init__(self, name='motor'):
        Initializes the MotorBuilder object with a given name.
    get_states(self) -> dict:
        Returns a dictionary of the subsystem's states, where the keys are the names of the state variables, and the values are dictionaries that contain the units for the state variable and any additional keyword arguments required by OpenMDAO for the state variable.
    get_linked_variables(self) -> list:
        Links voltage input from the battery subsystem
    build_pre_mission(self) -> openmdao.core.System:
        Builds an OpenMDAO system for the pre-mission computations of the subsystem.
    build_mission(self, num_nodes, aviary_inputs) -> openmdao.core.System:
        Builds an OpenMDAO system for the mission computations of the subsystem.
    get_constraints(self) -> dict:
        Returns a dictionary of constraints for the motor subsystem, where the keys are the names of the variables to be constrained, and the values are dictionaries that contain the lower and upper bounds for the constraint and any additional keyword arguments accepted by Dymos for the constraint.
    get_design_vars(self) -> dict:
        Returns a dictionary of design variables for the motor subsystem, where the keys are the names of the design variables, and the values are dictionaries that contain the units for the design variable, the lower and upper bounds for the design variable, and any additional keyword arguments required by OpenMDAO for the design variable.
    get_parameters(self) -> dict:
        Returns a dictionary of fixed values for the motor subsystem, where the keys are the names of the fixed values, and the values are dictionaries that contain the fixed value for the variable, the units for the variable, and any additional keyword arguments required by OpenMDAO for the variable.
    get_initial_guesses(self) -> dict:
        Returns a dictionary of initial guesses for the motor subsystem, where the keys are the names of the initial guesses, and the values are dictionaries that contain the initial guess value, the type of variable (state or control), and any additional keyword arguments required by OpenMDAO for the variable.
    get_mass_names(self) -> list:
        Returns a list of names for the motor subsystem mass.
    preprocess_inputs(self, aviary_inputs) -> aviary_inputs:
        No preprocessing needed for the motor subsystem.
    '''

    def __init__(self, name='motor', include_constraints=True):
        self.include_constraints = include_constraints
        super().__init__(name, meta_data=ExtendedMetaData)

    def get_states(self):
        '''
        Return a dictionary of states for the motor subsystem.

        Returns
        -------
        states : dict
            A dictionary where the keys are the names of the state variables
            and the values are dictionaries with the following keys:

            - 'units': str
                The units for the state variable.
            - any additional keyword arguments required by OpenMDAO for the state
              variable.
        '''
        states_dict = {}

        return states_dict

    def get_linked_variables(self):
        '''
        Return the list of linked variables for the motor subsystem.
        '''
        return []

    def build_pre_mission(self, aviary_inputs):
        '''
        Build an OpenMDAO system for the pre-mission computations of the subsystem.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in
            the pre-mission (formerly statics) part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        '''
        return MotorPreMission(aviary_inputs=aviary_inputs)

    def build_mission(self, num_nodes, aviary_inputs):
        '''
        Build an OpenMDAO system for the mission computations of the subsystem.

        Returns
        -------
        mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen
            during the mission. This includes time-dependent states that are
            being integrated as well as any other variables that vary during
            the mission.
        '''
        return MotorMission(num_nodes=num_nodes, aviary_inputs=aviary_inputs)

    def get_constraints(self):
        '''
        Return a dictionary of constraints for the motor subsystem.

        Returns
        -------
        constraints : dict
            A dictionary where the keys are the names of the variables to be constrained
            and the values are dictionaries are any accepted by Dymos for the
            constraint.

        Description
        -----------
        This method returns a dictionary of constraints for the motor subsystem.
        '''
        if self.include_constraints:
            # TBD
            constraints = {
                Dynamic.Mission.Motor.TORQUE_CON: {
                    'upper': 0.0,
                    'type': 'path'
                }
            }
        else:
            constraints = {}

        return constraints

    def get_design_vars(self):
        '''
        Return a dictionary of design variables for the battery subsystem.

        Returns
        -------
        design_vars : dict
            A dictionary where the keys are the names of the design variables
            and the values are dictionaries with the following keys:

            - 'units': str
                The units for the design variable
            - 'lower': float or None
                The lower bound for the design variable
            - 'upper': float or None
                The upper bound for the design variable
            - any additional keyword arguments required by OpenMDAO for the design
              variable
        '''

        DVs = {
            # TBD do we need this?
            Dynamic.Mission.THROTTLE: {
                'units': 'unitless',
                'lower': 0.0,
                'upper': 1.0
            },
            Aircraft.Engine.SCALE_FACTOR: {
                'units': 'unitless',
                'lower': 0.001,
                'upper': None
            },
            Aircraft.Motor.RPM: {
                'units': 'rpm',
                'lower': 0.1,
                'upper': 20000
            },
        }

        return DVs

    def get_parameters(self):
        '''
        Return a dictionary of fixed values exposed to the phases for the motor subsystem.

        Returns
        -------
        parameter_info : dict
            A dictionary where the keys are the names of the fixed values
            and the values are dictionaries with the following keys:

            - 'value': float or None
                The fixed value for the variable.
            - 'units': str or None
                The units for the variable.
            - any additional keyword arguments required by OpenMDAO for the variable.
        '''

        parameters_dict = {}

        return parameters_dict

    def get_initial_guesses(self):
        '''
        Return a dictionary of initial guesses for the motor subsystem.

        Returns
        -------
        initial_guesses : dict
            A dictionary where the keys are the names of the initial guesses
            and the values are dictionaries with any additional keyword
            arguments required by OpenMDAO for the variable.
        '''

        initial_guess_dict = {
            Dynamic.Mission.THROTTLE: {
                'units': 'unitless',
                'type': 'control',
                'val': 0.5,
            },
            Aircraft.Engine.SCALE_FACTOR: {
                'units': 'unitless',
                'type': 'parameter',
                'val': 1.0,
            },
            Aircraft.Motor.RPM: {
                'units': 'rpm',
                'type': 'parameter',
                'val': 4000.0,  # based on our map
            },
        }

        return initial_guess_dict

    def get_mass_names(self):
        '''
        Return a list of names for the motor subsystem.

        Returns
        -------
        mass_names : list
            A list of names for the motor subsystem.
        '''
        return [Aircraft.Motor.MASS]

    def preprocess_inputs(self, aviary_inputs):
        '''
        Preprocess the inputs for the motor subsystem.

        Description
        -----------
        This method preprocesses the inputs for the motor subsystem.
        In this case, it sets the values motor performance based on the motor cell type.
        '''

        return aviary_inputs

    def get_outputs(self):
        '''
        Return a list of output names for the motor subsystem.

        Returns
        -------
        outputs : list
            A list of variable names for the motor subsystem.
        '''

        return [Dynamic.Mission.Motor.TORQUE,
                Dynamic.Mission.Motor.SHAFT_POWER,
                Dynamic.Mission.Motor.ELECTRIC_POWER,
                Mission.Motor.ELECTRIC_ENERGY]
