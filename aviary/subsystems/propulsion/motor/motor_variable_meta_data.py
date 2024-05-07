import aviary.api as av
from aviary.subsystems.propulsion.motor.motor_variables import Aircraft, Dynamic, Mission


ExtendedMetaData = av.CoreMetaData

##### MOTOR VALUES #####

av.add_meta_data(
    Aircraft.Motor.COUNT,
    units=None,
    desc="Number of motors",
    default_value=1,
    option=True,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Motor.MASS,
    units="kg",
    desc="Total motor mass (considers number of motors)",
    default_value=1.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Motor.RPM,
    units="rpm",
    desc="Motor RPM",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Motor.TORQUE_MAX,
    units="N*m",
    desc="Max torque value that can be output from a single motor",
    meta_data=ExtendedMetaData
)

##### MOTOR MISSION VALUES #####

av.add_meta_data(
    Dynamic.Mission.Motor.EFFICIENCY,
    units=None,
    desc="Motor efficiency",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Dynamic.Mission.Motor.ELECTRIC_POWER,
    units="kW",
    desc="Power used by all the motors combined",
    default_value=None,
    meta_data=ExtendedMetaData
)

# av.add_meta_data(
#     Mission.Motor.ELECTRIC_ENERGY,
#     units="kW*h",
#     desc="Energy used by all the motors combined throughout the whole phase/mission",
#     default_value=None,
#     meta_data=ExtendedMetaData
# )

av.add_meta_data(
    Dynamic.Mission.Motor.SHAFT_POWER,
    units="kW",
    desc="Power output from a single motor",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Dynamic.Mission.Motor.TORQUE,
    units="N*m",
    desc="Motor torque",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Dynamic.Mission.Prop.TORQUE,
    units="N*m",
    desc="Torque output to a single propellar shaft from a motor/turbine/gearbox",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Dynamic.Mission.Motor.TORQUE_CON,
    units="N*m",
    desc="Motor torque constraint to ensure torque in mission is less than torque_max",
    default_value=None,
    meta_data=ExtendedMetaData
)
