"""


In this example, Two seperate aviary problems are instantiated using the typical aviaryProblem calls like load_inputs,
check_and_preprocess_payload, etc.. Once those problems are setup and all of their phases are linked together,
we copy those problems as group into a super_problem. We then promote GROSS_MASS, RANGE, SPAN, and wing AREA from each
of those sub-groups (group1 and group2) up to the super_probem so the optimizer can control them. The fuel_burn results
from each of the group1 and group2 dymos missions are summed and weighted to create the objective function the 
optimizer sees.

"""
import copy as copy
from aviary.examples.example_phase_info_N3CC import phase_info
from aviary.variable_info.variables import Mission, Aircraft
from aviary.variable_info.enums import ProblemType
import aviary.api as av
import openmdao.api as om
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import dymos as dm
import warnings
import sys
from aviary.subsystems.mass.flops_based.furnishings import TransportFurnishingsGroupMass
from aviary.api import SubsystemBuilderBase
from aviary.validation_cases.validation_tests import get_flops_inputs

# fly the same mission twice with two different passenger loads
phase_info_primary = copy.deepcopy(phase_info)
phase_info_heavy = copy.deepcopy(phase_info)

# instantiating a problem just so we can get the right aviary_inputs
prob_tmp = av.AviaryProblem()
aviary_inputs_primary = prob_tmp.load_inputs(
    get_flops_inputs('N3CC'), phase_info_primary)
aviary_inputs_primary.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 20348, 'lbm')

# Always choose a gross mass guess that is LOWER than the final expected GROSS_MASS
# Explanation: GROSS_MASS_constraint exists in methods_for_level2.py. This constraint
# forces the Design.GROSS_MASS to be larger or equal to the largest Summary.GROSS_MASS.
# This constraint only becomes active when the initial guess on Design.GROSS_MASS is
# small. When guessing a Design.GROSS_MASS that is larger, SLSQP does not always work
# hard enough to minimize that value and it's feasible so it stops the opt before it should.
aviary_inputs_primary.set_val(Mission.Design.GROSS_MASS, 120734., 'lbm')

aviary_inputs_heavy = copy.deepcopy(aviary_inputs_primary)
aviary_inputs_heavy.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 40348, 'lbm')


class MultiMissionProblem(om.Problem):
    def __init__(self, aviary_values, phase_infos, weights):
        super().__init__()
        self.num_missions = len(aviary_values)
        # phase infos and aviary_values length must match - this maybe unnecessary if
        # different aviary_values (payloads) fly same mission (say pax vs cargo)
        # or if same payload flies 2 different missions (altitude/mach differences)
        if self.num_missions != len(phase_infos):
            raise Exception("Length of aviary_values and phase_infos must be the same!")

        # if fewer weights than aviary_values are provided, assign equal weights for all aviary_values
        if len(weights) < self.num_missions:
            weights = [1]*self.num_missions
        # if more weights than aviary_values, raise exception
        elif len(weights) > self.num_missions:
            raise Exception("Length of weights cannot exceed length of aviary_values!")
        self.weights = weights
        self.phase_infos = phase_infos

        self.group_prefix = 'group'
        self.probs = []
        self.fuel_vars = []
        self.phases = {}
        # define individual aviary problems
        for i, (aviary_values, phase_info) in enumerate(zip(aviary_values, phase_infos)):
            prob = av.AviaryProblem()
            prob.load_inputs(aviary_values, phase_info)
            prob.check_and_preprocess_inputs()
            prob.add_pre_mission_systems()
            prob.add_phases()
            prob.add_post_mission_systems()
            prob.link_phases()

            # alternate prevents use of equality constraint b/w design and summary gross mass
            prob.problem_type = ProblemType.MULTI_MISSION
            prob.add_design_variables()

            # Add Time Constraint
            # traj = prob.model._get_subsystem('traj')
            # descent = traj.phases._get_subsystem('descent')
            # descent.add_boundary_constraint('time', loc='final', upper=480, units='min')

            self.probs.append(prob)
            # phase names for each traj (can be used later to make plots/print outputs)
            self.phases[f"{self.group_prefix}_{i}"] = list(prob.traj._phases.keys())

            # design range and gross mass are promoted, these are Max Range/Max Takeoff Mass
            # and must be the same for each aviary problem. Subsystems within aviary are sized
            # using these - empty mass is same across all aviary problems.
            # the fuel objective is also promoted since that's used in the compound objective
            promoted_name = f"{self.group_prefix}_{i}_fuelobj"
            self.fuel_vars.append(promoted_name)
            self.model.add_subsystem(
                self.group_prefix + f'_{i}', prob.model,
                promotes_inputs=[Mission.Design.GROSS_MASS,
                                 Mission.Design.RANGE,
                                 #  Aircraft.Wing.SPAN,
                                 #  Aircraft.Wing.AREA
                                 ],
                promotes_outputs=[(Mission.Objectives.FUEL, promoted_name)])

    def add_design_variables(self):
        self.model.add_design_var(Mission.Design.GROSS_MASS,
                                  lower=10., upper=900e3, units='lbm')
        # self.model.add_design_var(Aircraft.Wing.SPAN, lower=100., upper=500., units='ft')
        # self.model.add_design_var(Aircraft.Wing.AREA, lower=10.,
        #                           upper=1e6, units='ft**2')

    def add_driver(self):
        self.driver = om.pyOptSparseDriver()
        self.driver.options["optimizer"] = "SLSQP"
        self.driver.declare_coloring()
        # linear solver causes nan entry error for landing to takeoff mass ratio param
        # self.model.linear_solver = om.DirectSolver()

    # def add_mux(self, fuel_vars):
    #     # Add a mux to turn the fuel_burn into a vector

    #     fuel_mux = self.model.add_subsystem(
    #         name='fuel_mux', subsys=om.MuxComp(vec_size=len(fuel_vars)))

    #     for name in fuel_vars:
    #         fuel_mux.add_var(name, shape=(1,), axis=1, units='kg')

    def add_objective(self):
        # weights are normalized - e.g. for given weights 3:1, the normalized
        # weights are 0.75:0.25
        weights = [float(weight/sum(self.weights)) for weight in self.weights]
        weighted_str = "+".join([f"{fuelobj}*{weight}"
                                for fuelobj, weight in zip(self.fuel_vars, weights)])
        # weighted_str looks like: fuel_0 * weight[0] + fuel_1 * weight[1]
        # note that the fuel objective itself is the base aviary fuel objective
        # which is also a function of climb time becuse climb is not very sensitive to fuel

        # adding compound execComp to super problem
        self.model.add_subsystem('compound_fuel_burn_objective', om.ExecComp(
            "compound = "+weighted_str, has_diag_partials=True), promotes=["compound"])
        self.model.add_objective('compound')

    def setup_wrapper(self):
        """Wrapper for om.Problem setup with warning ignoring and setting options"""
        for prob in self.probs:
            prob.model.options['aviary_options'] = prob.aviary_inputs
            prob.model.options['aviary_metadata'] = prob.meta_data
            prob.model.options['phase_info'] = prob.phase_info

        # Aviary's problem setup wrapper uses these ignored warnings to suppress
        # some warnings related to variable promotion. Replicating that here with
        # setup for the super problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", om.OpenMDAOWarning)
            warnings.simplefilter("ignore", om.PromotionWarning)
            self.setup(check='all')

    def run(self):
        self.model.set_solver_print(0)
        dm.run_problem(self, make_plots=False)

    def get_design_range(self):
        """Finds the longest mission and sets its range as the design range for all
            Aviary problems. Used within Aviary for sizing subsystems (avionics and AC)."""
        design_range = []
        for phase_info in self.phase_infos:
            design_range.append(phase_info['post_mission']
                                ['target_range'][0])  # TBD add units
        design_range_min = np.min(design_range)
        design_range_max = np.max(design_range)
        return design_range_max, design_range_min  # design_range_min

    def create_timeseries_plots(self, plotvars=[], show=True):
        """
        Temporary create plots manually because graphing won't work for dual-trajectories.
        Creates timeseries plots for any variables within timeseries. Specify variables
        and units by setting plotvars = [('altitude','ft')]. Any number of vars can be added.
        """
        plt.figure()
        for plotidx, (var, unit) in enumerate(plotvars):
            plt.subplot(int(np.ceil(len(plotvars)/2)), 2, plotidx+1)
            for i in range(self.num_missions):
                time = np.array([])
                yvar = np.array([])
                # this loop concatenates data from all phases
                for phase in self.phases[f"{self.group_prefix}_{i}"]:
                    rawt = self.get_val(
                        f"{self.group_prefix}_{i}.traj.{phase}.timeseries.time",
                        units='s')
                    rawy = self.get_val(
                        f"{self.group_prefix}_{i}.traj.{phase}.timeseries.{var}",
                        units=unit)
                    time = np.hstack([time, np.ndarray.flatten(rawt)])
                    yvar = np.hstack([yvar, np.ndarray.flatten(rawy)])
                plt.plot(time, yvar, linewidth=self.num_missions-i)
            plt.xlabel("Time (s)")
            plt.ylabel(f"{var.title()} ({unit})")
            plt.grid()
        plt.figlegend([f"Plane {i}" for i in range(self.num_missions)])
        if show:
            plt.show()

    def create_payload_range_plot(self, show=True):
        """Creates payload range diagram for the super problem. Appends a point for max payload
            and 0 range. """
        payloads = []
        ranges = []
        for i in range(self.num_missions):
            ref = f"{self.group_prefix}_{i}"
            payloads.append(
                self.get_val(
                    f"{ref}.{Aircraft.CrewPayload.CARGO_MASS}", units='lbm'))
            lastphase = self.phases[ref][-1]
            ranges.append(
                self.get_val(
                    f"{ref}.traj.{lastphase}.timeseries.distance",
                    units='nmi', indices=-1)[0])
        payloads, ranges = zip(*sorted(zip(payloads, ranges)))
        payloads, ranges = list(payloads), list(ranges)
        payloads.append(payloads[-1])
        ranges.append(0)
        plt.figure()
        plt.plot(ranges, payloads)
        plt.xlabel("Range (nmi)")
        plt.ylabel("Payload (lbm)")
        plt.grid()
        if show:
            plt.show()

    def print_vars(self, vars=[]):
        """Specify vars with name and unit in a tuple, e.g. vars = [ (Mission.Summary.FUEL_BURNED, 'lbm') ]"""

        print("\n\n=========================\n")
        print(f"{'':40}", end=': ')
        for i in range(self.num_missions):
            name = f"Mission {i}"
            print(f"{name:^30}", end='| ')
        print()
        for var, unit in vars:
            varname = f"Variable: {var.replace(':', '.').upper()}"
            print(f"{varname:40}", end=": ")
            for i in range(self.num_missions):
                val = self.get_val(f'group_{i}.{var}', units=unit)[0]
                printstatement = f"{val} ({unit})"
                print(f"{printstatement:^30}", end="| ")
            print()


def large_single_aisle_example(makeN2=False):
    aviary_values = [aviary_inputs_primary,
                     aviary_inputs_heavy]
    phase_infos = [phase_info_primary,
                   phase_info_heavy]
    optalt, optmach = False, False
    for phaseinfo in phase_infos:
        for key in phaseinfo.keys():
            if "user_options" in phaseinfo[key].keys():
                phaseinfo[key]["user_options"]["optimize_mach"] = optmach
                phaseinfo[key]["user_options"]["optimize_altitude"] = optalt

    # how much each mission should be valued by the optimizer, larger numbers = more significance
    weights = [9, 1]

    super_prob = MultiMissionProblem(aviary_values, phase_infos, weights)
    super_prob.add_driver()
    super_prob.add_design_variables()
    super_prob.add_objective()

    # set input default to prevent error, value doesn't matter since set val is used later
    super_prob.model.set_input_defaults(Mission.Design.RANGE, val=1.)
    super_prob.setup_wrapper()
    super_prob.set_val(Mission.Design.RANGE, super_prob.get_design_range()[0])

    for i, prob in enumerate(super_prob.probs):
        prob.set_initial_guesses(super_prob, super_prob.group_prefix+f"_{i}.")

    if makeN2:
        # TODO: Not sure we need this at all.
        from openmdao.api import n2
        from os.path import basename, dirname, join, abspath

        def createN2(fileref, prob):
            n2folder = join(dirname(abspath(__file__)), "N2s")
            n2(prob, outfile=join(n2folder,
                                  f"n2_{basename(fileref).split('.')[0]}.html"))

        createN2(__file__, super_prob)

    super_prob.run()
    printoutputs = [
        (Mission.Design.GROSS_MASS, 'lbm'),
        (Aircraft.Design.EMPTY_MASS, 'lbm'),
        (Mission.Summary.GROSS_MASS, 'lbm'),
        (Mission.Summary.FUEL_BURNED, 'lbm'),
        (Mission.Design.FUEL_MASS, 'lbm'),
        (Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
        (Aircraft.Wing.SPAN, 'ft'),
        (Aircraft.Wing.AREA, 'ft**2'),
        (Aircraft.LandingGear.MAIN_GEAR_MASS, 'lbm'),
        (Aircraft.LandingGear.NOSE_GEAR_MASS, 'lbm'),
        (Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 'unitless'),
        (Mission.Summary.CRUISE_MACH, 'unitless'),
        (Aircraft.Furnishings.MASS, 'lbm'),
        (Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, 'lbm')]
    super_prob.print_vars(vars=printoutputs)

    plotvars = [('altitude', 'ft'),
                ('mass', 'lbm'),
                ('drag', 'lbf'),
                ('distance', 'nmi'),
                ('throttle', 'unitless'),
                ('mach', 'unitless')]
    super_prob.create_timeseries_plots(plotvars=plotvars, show=False)

    super_prob.create_payload_range_plot(show=False)
    plt.show()

    return super_prob


if __name__ == '__main__':
    makeN2 = True if (len(sys.argv) > 1 and "n2" in sys.argv[1]) else False

    super_prob = large_single_aisle_example(makeN2=makeN2)

    # Uncomment the following lines to see mass breakdown details for each mission.
    # super_prob.model.group_1.list_vars(val=True, units=True, print_arrays=False)
    # super_prob.model.group_2.list_vars(val=True, units=True, print_arrays=False)
