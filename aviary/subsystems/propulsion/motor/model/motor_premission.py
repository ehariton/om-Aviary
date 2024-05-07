import openmdao.api as om

from aviary.subsystems.propulsion.motor.motor_variables import Mission, Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues
from aviary.subsystems.propulsion.motor.model.motor_map import MotorMap


class MotorPreMission(om.Group):
    """
    Calculate electric motor mass for a single motor
    """

    def initialize(self):
        self.options.declare(
            "aviary_inputs", types=AviaryValues,
            desc="collection of Aircraft/Mission specific options",
            default=None,
        )
        self.name = 'motor_premission'

    def setup(self):

        # Determine max torque of scaled motor

        # We create a set of default inputs for this group so that in pre-mission,
        # the group can be instantiated with only scale_factor as an input.
        # without inputs and it will return the max torque
        # based on the non-dimensional scale factor chosen by the optimizer.
        # The max torque is then used in pre-mission to determine weight of the system.
        self.set_input_defaults(Dynamic.Mission.THROTTLE, 1.0, units='unitless')

        # TBD I'm worried that the above code won't set these in pre-mission correctly
        self.add_subsystem('motor_map', MotorMap(num_nodes=1),
                           promotes_inputs=[Aircraft.Engine.SCALE_FACTOR,
                                            Dynamic.Mission.THROTTLE,
                                            Aircraft.Motor.RPM],
                           promotes_outputs=[(Dynamic.Mission.Motor.TORQUE,
                                              Aircraft.Motor.TORQUE_MAX)])

        # Motor mass relationship based on continuous torque rating for aerospace motors (Figure 10)
        # Propulsion Scaling Methods in the Era of Electric Flight - Duffy et. al.
        # AIAA Propulsion and Energy Forum, July 9-11, 2018
        self.add_subsystem('motor_mass',
                           om.ExecComp('motor_mass = (0.3151 * max_torque ** 0.748)',
                                       motor_mass={'val': 1.0, 'units': 'kg'},
                                       max_torque={'val': 1.0, 'units': 'N*m'}),
                           promotes_inputs=[('max_torque', Aircraft.Motor.TORQUE_MAX)],
                           promotes_outputs=[('motor_mass', Aircraft.Motor.MASS)])

        # TBD Gearbox mass calc goes here?
