
from aviary.variable_info.enums import ProblemType
from aviary.variable_info.variables import Mission, Aircraft, Settings
import aviary.api as av
from example_phase_info_N3CC import phase_info
from aviary.validation_cases.validation_tests import get_flops_inputs

prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
# we want a longer-range mission:

aviary_inputs = prob.load_inputs(get_flops_inputs('N3CC'), phase_info)

aviary_inputs.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 41348, 'lbm')
aviary_inputs.set_val(Settings.VERBOSITY, av.Verbosity.BRIEF)

# Preprocess inputs
prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver("SNOPT", max_iter=100)

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()

traj = prob.model._get_subsystem('traj')
descent = traj.phases._get_subsystem('descent')
descent.add_boundary_constraint('time', loc='final', upper=480, units='min')

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()
