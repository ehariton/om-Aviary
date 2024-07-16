from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.motor.model.motor_premission import GearboxPreMission
from aviary.subsystems.propulsion.motor.model.motor_mission import GearboxMission
from aviary.variable_info.variables import Aircraft, Dynamic


class GearboxBuilder(SubsystemBuilderBase):
    '''
    Define the builder for a single gearbox subsystem that provides methods to define the motor subsystem's states, design variables, fixed values, initial guesses, and mass names.

    It also provides methods to build OpenMDAO systems for the pre-mission and mission computations of the subsystem, to get the constraints for the subsystem, and to preprocess the inputs for the subsystem.

    This is meant to be computations for a single gearbox, so there is no notion of "num_motors" in this code.
    '''

    def __init__(self, name='gearbox', include_constraints=True):
        '''Initializes the MotorBuilder object with a given name.'''
        self.include_constraints = include_constraints
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        '''Builds an OpenMDAO system for the pre-mission computations of the subsystem.'''
        return MotorPreMission(aviary_inputs=aviary_inputs)

    def build_mission(self, num_nodes, aviary_inputs):
        '''Builds an OpenMDAO system for the mission computations of the subsystem.'''
        return MotorMission(num_nodes=num_nodes, aviary_inputs=aviary_inputs)

    def get_design_vars(self):
        '''
        Returns a dictionary of design variables for the motor subsystem, where the keys are the 
        names of the design variables, and the values are dictionaries that contain the units for 
        the design variable, the lower and upper bounds for the design variable, and any 
        additional keyword arguments required by OpenMDAO for the design variable.
        '''

        DVs = {
            Aircraft.Engine.Gearbox.GEAR_RATIO: {
                'units': None,
                'lower': 1.0,
                'upper': 20.0,
            }
        }

        return DVs

    def get_parameters(self):
        '''
        Returns a dictionary of fixed values for the gearbox subsystem, where the keys are the names 
        of the fixed values, and the values are dictionaries that contain the fixed value for the 
        variable, the units for the variable, and any additional keyword arguments required by 
        OpenMDAO for the variable.

        Returns
        -------
        parameters : list
        A list of names for the gearbox subsystem.
        '''
        parameters = {
            Aircraft.Engine.Gearbox.EFFICIENCY: {
                'val': 0.98,
                'units': None,
            }
        }

        return parameters

    def get_mass_names(self):
        '''
        Return a list of names for the gearbox subsystem.

        Returns
        -------
        mass_names : list
            A list of names for the gearbox subsystem.
        '''
        return [Aircraft.Engine.Gearbox.MASS]

    def get_outputs(self):
        '''
        Return a list of output names for the gearbox subsystem.

        Returns
        -------
        outputs : list
            A list of variable names for the gearbox subsystem.
        '''

        return [
            Dynamic.Mission.RPM_GEAR,
            Dynamic.Mission.SHAFT_POWER_GEAR,
            Dynamic.Mission.SHAFT_POWER_GEAR_MAX,
            Dynamic.Mission.TORQUE_GEAR,
        ]
