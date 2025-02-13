import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic


class DynamicPressure(om.ExplicitComponent):
    """
    Compute dynamic pressure as
    Dynamic.Mission.DYNAMIC_PRESSURE = 0.5 * gamma * P * M**2
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare(
            'gamma', default=1.4, desc='Ratio of specific heats for air.')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            Dynamic.Atmosphere.STATIC_PRESSURE,
            np.ones(nn),
            units='lbf/ft**2',
            desc='Static pressure at each evaulation point.',
        )

        self.add_input(
            Dynamic.Atmosphere.MACH,
            np.ones(nn),
            units='unitless',
            desc='Mach at each evaulation point.',
        )

        self.add_output(
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            val=np.ones(nn),
            units='lbf/ft**2',
            desc='pressure caused by fluid motion',
        )

    def setup_partials(self):
        nn = self.options['num_nodes']

        rows_cols = np.arange(nn)

        self.declare_partials(
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            [Dynamic.Atmosphere.STATIC_PRESSURE, Dynamic.Atmosphere.MACH],
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs):
        gamma = self.options['gamma']
        P = inputs[Dynamic.Atmosphere.STATIC_PRESSURE]
        M = inputs[Dynamic.Atmosphere.MACH]

        outputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE] = 0.5 * gamma * P * M**2

    def compute_partials(self, inputs, partials):
        gamma = self.options['gamma']
        P = inputs[Dynamic.Atmosphere.STATIC_PRESSURE]
        M = inputs[Dynamic.Atmosphere.MACH]

        partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.MACH] = (
            gamma * P * M
        )
        partials[
            Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.STATIC_PRESSURE
        ] = (0.5 * gamma * M**2)
