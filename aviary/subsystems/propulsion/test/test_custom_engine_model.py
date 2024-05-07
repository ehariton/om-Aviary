import unittest

import dymos as dm
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.variable_info.options import get_option_defaults
from aviary.utils.functions import get_path
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.subsystems.propulsion.motor.motor_variables import Aircraft, Dynamic, Mission


class PreMissionEngine(om.Group):
    def setup(self):
        self.add_subsystem('dummy_comp', om.ExecComp(
            'y=x**2', x={'units': 'm', 'val': 2.}, y={'units': 'm**2'}), promotes=['*'])


class SimpleEngine(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # add inputs and outputs to interpolator
        self.add_input(Dynamic.Mission.MACH,
                       shape=nn,
                       units='unitless',
                       desc='Current flight Mach number')
        self.add_input(Dynamic.Mission.ALTITUDE,
                       shape=nn,
                       units='ft',
                       desc='Current flight altitude')
        self.add_input(Dynamic.Mission.THROTTLE,
                       shape=nn,
                       units='unitless',
                       desc='Current engine throttle')
        self.add_input('different_throttle',
                       shape=nn,
                       units='unitless',
                       desc='Little bonus throttle for testing')
        self.add_input('y',
                       units='m**2',
                       desc='Dummy variable for bus testing')

        self.add_output(Dynamic.Mission.THRUST,
                        shape=nn,
                        units='lbf',
                        desc='Current net thrust produced (scaled)')
        self.add_output(Dynamic.Mission.THRUST_MAX,
                        shape=nn,
                        units='lbf',
                        desc='Current net thrust produced (scaled)')
        self.add_output(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        shape=nn,
                        units='lbm/s',
                        desc='Current fuel flow rate (scaled)')
        self.add_output(Dynamic.Mission.ELECTRIC_POWER,
                        shape=nn,
                        units='W',
                        desc='Current electric energy rate (scaled)')
        self.add_output(Dynamic.Mission.NOX_RATE,
                        shape=nn,
                        units='lbm/s',
                        desc='Current NOx emission rate (scaled)')
        self.add_output(Dynamic.Mission.TEMPERATURE_ENGINE_T4,
                        shape=nn,
                        units='degR',
                        desc='Current turbine exit temperature')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        combined_throttle = inputs[Dynamic.Mission.THROTTLE] + \
            inputs['different_throttle']

        # calculate outputs
        outputs[Dynamic.Mission.THRUST] = 10000. * combined_throttle
        outputs[Dynamic.Mission.THRUST_MAX] = 10000.
        outputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE] = -10. * combined_throttle
        outputs[Dynamic.Mission.TEMPERATURE_ENGINE_T4] = 2800.


class SimpleTestEngine(EngineModel):
    def __init__(self, name='engine', options=None):
        aviary_inputs = AviaryValues()
        super().__init__(name, options=aviary_inputs)

    def build_pre_mission(self, aviary_inputs=AviaryValues()):
        return PreMissionEngine()

    def build_mission(self, num_nodes, aviary_inputs):
        return SimpleEngine(num_nodes=num_nodes)

    def get_controls(self):
        controls_dict = {
            "different_throttle": {'units': 'unitless', 'lower': 0., 'upper': 0.1},
        }
        return controls_dict

    def get_bus_variables(self):
        bus_dict = {
            "y": {
                "mission_name": "y",
                "units": "m**2",
            },
        }
        return bus_dict

    def get_initial_guesses(self):
        initial_guesses_dict = {
            "different_throttle": {
                "val": 0.05,
                "units": "unitless",
                "type": "control",
            }
        }
        return initial_guesses_dict


@use_tempdirs
class CustomEngineTest(unittest.TestCase):
    def test_custom_engine(self):

        phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [],
                'optimize_mass': True,
            },
            'cruise': {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 2,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.72, "unitless"),
                    "final_mach": (0.72, "unitless"),
                    "mach_bounds": ((0.7, 0.74), "unitless"),
                    "initial_altitude": (35000.0, "ft"),
                    "final_altitude": (35000.0, "ft"),
                    "altitude_bounds": ((23000.0, 38000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": False,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 0.0), "min"),
                    "duration_bounds": ((10., 30.), "min"),
                },
                "initial_guesses": {"time": ([0, 30], "min")},
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [],
            }
        }

        prob = AviaryProblem(reports=False)

        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs("models/test_aircraft/aircraft_for_bench_GwFm.csv",
                         phase_info, engine_builder=SimpleTestEngine())

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SLSQP", verbosity=Verbosity.QUIET)

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_initial_guesses()

        prob.final_setup()

        # check that the different throttle initial guess has been set correctly
        initial_guesses = prob.get_val(
            'traj.phases.cruise.controls:different_throttle')[0]
        assert_near_equal(float(initial_guesses), 0.05)

        # and run mission, and dynamics
        dm.run_problem(prob, run_driver=True, simulate=False, make_plots=False)

        tol = 1.e-4

        assert_near_equal(float(prob.get_val('traj.cruise.rhs_all.y')), 4., tol)


@unittest.skip("Skipping until engines are no longer required to always output all values")
@use_tempdirs
class TurbopropTest(unittest.TestCase):
    def test_turboprop(self):
        phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [],
                'optimize_mass': True,
            },
            'cruise': {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 2,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.76, "unitless"),
                    "final_mach": (0.76, "unitless"),
                    "mach_bounds": ((0.7, 0.78), "unitless"),
                    "initial_altitude": (35000.0, "ft"),
                    "final_altitude": (35000.0, "ft"),
                    "altitude_bounds": ((23000.0, 38000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": False,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 0.0), "min"),
                    "duration_bounds": ((30., 60.), "min"),
                },
                "initial_guesses": {"time": ([0, 30], "min")},
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [],
            }
        }

        engine_filepath = get_path('models/engines/turboprop_4465hp.deck')
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.DATA_FILE, engine_filepath)
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        options.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10, units='ft')

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES,
                        val=4, units='unitless')

        engine = TurbopropModel(options=options)

        prob = AviaryProblem(reports=True)

        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs("models/test_aircraft/aircraft_for_bench_FwFm.csv",
                         phase_info, engine_builder=engine)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SLSQP", max_iter=20)

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_initial_guesses()

        prob.set_val(
            f'traj.cruise.rhs_all.{Aircraft.Design.MAX_TIP_SPEED}', 710., units='ft/s')
        prob.set_val(
            f'traj.cruise.rhs_all.{Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED}', 0.915, units='unitless')
        prob.set_val(
            f'traj.cruise.rhs_all.{Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR}', 150., units='unitless')
        prob.set_val(
            f'traj.cruise.rhs_all.{Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT}', 0.5, units='unitless')

        prob.set_solver_print(level=0)

        # and run mission, and dynamics
        dm.run_problem(prob, run_driver=True, simulate=False, make_plots=True)


@unittest.skip("Skipping until engines are no longer required to always output all values")
#@use_tempdirs
class ElectropropTest(unittest.TestCase):
    def test_electroprop(self):
        phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [],
                'optimize_mass': True,
            },
            'cruise': {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 2,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.76, "unitless"),
                    "final_mach": (0.76, "unitless"),
                    "mach_bounds": ((0.7, 0.78), "unitless"),
                    "initial_altitude": (35000.0, "ft"),
                    "final_altitude": (35000.0, "ft"),
                    "altitude_bounds": ((23000.0, 38000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": False,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 0.0), "min"),
                    "duration_bounds": ((30., 60.), "min"),
                },
                "initial_guesses": {"time": ([0, 30], "min")},
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [],
            }
        }

        options = get_option_defaults()
        # options.set_val(Aircraft.Engine.DATA_FILE, engine_filepath)
        options.set_val(Aircraft.Motor.COUNT, 2)
        options.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10, units='ft')

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES,
                        val=4, units='unitless')

        engine = TurbopropModel(
            options=options,
            shaft_power_model=MotorBuilder(),
            propeller_model='hamilton_standard')

        prob = AviaryProblem(reports=True)

        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs("models/test_aircraft/aircraft_for_bench_FwFm.csv",
                         phase_info,
                         engine_builder=engine)

        prob.aviary_inputs.set_val(Aircraft.Motor.COUNT, 2)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SLSQP", max_iter=20)

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_initial_guesses()

        prob.set_val(
            f'traj.cruise.rhs_all.{Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR}', 150., units='unitless')
        prob.set_val(
            f'traj.cruise.rhs_all.{Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT}', 0.5, units='unitless')

        prob.set_solver_print(level=0)

        # and run mission, and dynamics
        dm.run_problem(prob, run_driver=True, simulate=False, make_plots=True)


if __name__ == '__main__':
    # unittest.main()
    ElectropropTest().test_electroprop()
