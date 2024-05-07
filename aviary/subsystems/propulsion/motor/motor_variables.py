from aviary.variable_info.variables import Aircraft as av_Aircraft
from aviary.variable_info.variables import Dynamic as av_Dynamic
from aviary.variable_info.variables import Mission as av_Mission

# ---------------------------
# Aircraft data hierarchy
# ---------------------------


class Aircraft(av_Aircraft):

    class Motor:
        COUNT = "aircraft:motor:count"
        MASS = "aircraft:motor:mass"
        RPM = "aircraft:motor:rpm"
        TORQUE_MAX = "aircraft:motor:torque_max"

    class Gearbox:
        GEAR_RATIO = "aircraft:gearbox:ratio"
        MASS = "aircraft:gearbox:mass"
        TORQUE_MAX = "aircraft:gearbox:torque_max"

# ---------------------------
# Mission data hierarchy
# ---------------------------


class Dynamic(av_Dynamic):

    class Mission(av_Dynamic.Mission):

        class Motor:
            EFFICIENCY = "dynamic:mission:motor:efficiency"
            ELECTRIC_POWER = "dynamic:mission.motor.electric_power"
            SHAFT_POWER = "dynamic:mission:motor:power_out"
            TORQUE = "dynamic:mission:motor:torque"
            TORQUE_CON = "Dynamic.Mission.Motor.TORQUE_CON"

        class Gearbox():
            EFFICIENCY = "dynamic:mission:gearbox:efficiency"
            SHAFT_POWER_IN = "dynamic:mission:gearbox:power_in"
            SHAFT_POWER_OUT = "dynamic:mission:gearbox:power_out"
            TORQUE_IN = "dynamic:mission:gearbox:torque_in"
            TORQUE_OUT = "dynamic:mission:gearbox:torque_out"

        class Prop(av_Dynamic.Mission.Prop):
            TORQUE = "dynamic:mission:prop:torque"


class Mission(av_Mission):

    class Motor:
        ELECTRIC_ENERGY = "mission.motor.electric_energy"
