import openmdao.api as om
import numpy as np

from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues


class GearboxPreMission(om.Group):
    """
    Calculate gearbox mass for a single gearbox
    """

    def initialize(self):
        self.options.declare(
            "aviary_inputs", types=AviaryValues,
            desc="collection of Aircraft/Mission specific options",
            default=None,
        )
        self.name = 'gearbox_premission'

    def setup(self):

        self.add_subsystem('gearbox_PRM',
                           om.ExecComp('RPM_out = gear_ratio * RPM_in',
                                       RPM_out={'val': 0.0, 'units': 'rpm'},
                                       gear_ratio={'val': 1.0, 'units': None},
                                       RPM_in={'val': 0.0, 'units': 'rpm'}),
                           promotes_inputs=[('RPM_in', Dynamic.Mission.RPM),  # 'Rotational rate of shaft, per engine.'
                                            ('gear_ratio', Aircraft.Engine.Gearbox.GEAR_RATIO)],
                           promotes_outputs=['RPM_out'])

        # Gearbox mass from "An N+3 Technolgoy Level Reference Propulsion System" by Scott Jones, William Haller, and Michael Tong
        # NASA TM 2017-219501
        self.add_subsystem('gearbox_mass',
                           om.ExecComp('gearbox_mass = (power / RPM_out)^(0.75) * (RPM_in / RPM_out)^(0.15)',
                                       gearbox_mass={'val': 0.0, 'units': 'lb'},
                                       power={'val': 0.0, 'units': 'hp'},
                                       RPM_out={'val': 0.0, 'units': 'rpm'},
                                       RPM_in={'val': 0.0, 'units': 'rpm'},),
                           promotes_inputs=[('power', Dynamic.Mission.SHAFT_POWER_MAX),  # 'The maximum possible shaft power currently producible, per engine'
                                            'RPM_out', ('RPM_in', Dynamic.Mission.RPM)],
                           promotes_outputs=[('gearbox_mass', Aircraft.Engine.Gearbox.MASS)])
