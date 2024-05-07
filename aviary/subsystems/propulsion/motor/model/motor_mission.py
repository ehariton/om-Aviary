import numpy as np

import openmdao.api as om
from aviary.utils.aviary_values import AviaryValues

from aviary.subsystems.propulsion.motor.motor_variables import Dynamic, Aircraft, Mission
from aviary.subsystems.propulsion.motor.model.motor_map import MotorMap


class MotorMission(om.Group):

    '''
    Calculates the mission performance (ODE) of a single electric motor.
    '''

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            'aviary_inputs', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.name = 'motor_mission'

    def setup(self):
        n = self.options["num_nodes"]

        ivc = om.IndepVarComp()
        ivc.add_output(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                       val=np.zeros(n), units='kg/s')
        ivc.add_output(Dynamic.Mission.ELECTRIC_POWER, val=np.zeros(n), units='kW')
        ivc.add_output(Dynamic.Mission.NOX_RATE, val=np.zeros(n), units='kg/s')
        self.add_subsystem('ivc', ivc, promotes=['*'])

        motor_group = om.Group()

        motor_group.add_subsystem('motor_map', MotorMap(num_nodes=n),
                                  promotes_inputs=[Dynamic.Mission.THROTTLE,
                                                   Aircraft.Engine.SCALE_FACTOR,
                                                   Aircraft.Motor.RPM],
                                  promotes_outputs=[Dynamic.Mission.Motor.TORQUE,
                                                    Dynamic.Mission.Motor.EFFICIENCY])

        motor_group.add_subsystem('power_comp',
                                  om.ExecComp('P = T * pi * RPM / 30',
                                              P={'val': np.ones(n), 'units': 'kW'},
                                              T={'val': np.ones(n), 'units': 'kN*m'},
                                              RPM={'val': np.ones(n), 'units': 'rpm'}),
                                  promotes_inputs=[("T", Dynamic.Mission.Motor.TORQUE),
                                                   ("RPM", Aircraft.Motor.RPM)],
                                  promotes_outputs=[("P", Dynamic.Mission.Motor.SHAFT_POWER)])

        motor_group.add_subsystem('energy_comp',
                                  om.ExecComp('P_elec = P / eff',
                                              P={'val': np.ones(n), 'units': 'kW'},
                                              P_elec={'val': np.ones(n), 'units': 'kW'},
                                              eff={'val': np.ones(n), 'units': None}),
                                  promotes_inputs=[("P", Dynamic.Mission.Motor.SHAFT_POWER),
                                                   ("eff", Dynamic.Mission.Motor.EFFICIENCY)],
                                  promotes_outputs=[("P_elec", Dynamic.Mission.Motor.ELECTRIC_POWER)])

        motor_group.add_subsystem('torque_con',
                                  om.ExecComp('torque_con = torque_max - torque_mission',
                                              torque_con={'val': np.ones(
                                                  n), 'units': 'kN*m'},
                                              torque_max={'val': 1.0, 'units': 'kN*m'},
                                              torque_mission={'val': np.ones(n), 'units': 'kN*m'}),
                                  promotes_inputs=[('torque_mission', Dynamic.Mission.Motor.TORQUE),
                                                   ('torque_max', Aircraft.Motor.TORQUE_MAX)],
                                  promotes_outputs=[('torque_con', Dynamic.Mission.Motor.TORQUE_CON)])

        self.add_subsystem('motor_group', motor_group,
                           promotes_inputs=['*'],
                           promotes_outputs=[
                               '*',
                               (Dynamic.Mission.Motor.TORQUE, Dynamic.Mission.Prop.TORQUE),
                               (Dynamic.Mission.Motor.SHAFT_POWER,
                                Dynamic.Mission.Prop.SHAFT_POWER)
                           ])
