import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

from iiwa_batter import NUM_JOINTS, PACKAGE_ROOT, PITCH_DT
from iiwa_batter.physics import (
    PITCH_START_X,
    PITCH_START_Z,
    PLATE_OFFSET_Y,
    STRIKE_ZONE_Z,
)

FLIGHT_TIME_MULTIPLE = (
    1.04  # Extra time to allow the ball to pass through the strike zone
)


def make_model_directive(dt=PITCH_DT, model_urdf="iiwa14_limitless"):
    # We're pitchin the ball from +x to -x
    # The robot is sitting next to the origin

    # Rotates the base towards the plate just a little so the first joint doesn't max out before crossing the strike zone
    base_rotation_deg = np.rad2deg(np.pi / 4)

    model_directive = f"""
directives:
- add_model:
    name: iiwa
    file: file://{PACKAGE_ROOT}/assets/{model_urdf}.urdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0]
        iiwa_joint_3: [0]
        iiwa_joint_4: [0]
        iiwa_joint_5: [0]
        iiwa_joint_6: [0]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::base
    X_PC:
        rotation: !Rpy
            deg: [0, 0, {base_rotation_deg}]
        translation: [0, {PLATE_OFFSET_Y}, 0]
- add_model:
    name: bat
    file: file://{PACKAGE_ROOT}/assets/bat.sdf
- add_weld:
    parent: iiwa::iiwa_link_ee
    child: bat::base
    X_PC:
        translation: [0.04, 0, 0.3]
- add_model:
    name: sweet_spot
    file: file://{PACKAGE_ROOT}/assets/sweet_spot.sdf
- add_weld:
    parent: bat::base
    child: sweet_spot::base
    X_PC:
        translation: [0, 0, 0.4]
- add_model:
    name: ball
    file: file://{PACKAGE_ROOT}/assets/ball.sdf
    default_free_body_pose:
        base:
            translation: [{PITCH_START_X}, 0, {PITCH_START_Z}]
- add_model:
    name: strike_zone
    file: file://{PACKAGE_ROOT}/assets/strike_zone.sdf
- add_weld:
    parent: world
    child: strike_zone::base
    X_PC:
        translation: [0, 0, {STRIKE_ZONE_Z}]

model_drivers:
    iiwa: !IiwaDriver
      hand_model_name: wsg
      control_mode: torque_only

plant_config:
    contact_model: "hydroelastic"
    discrete_contact_approximation: "lagged"
    time_step: {dt}
    penetration_allowance: 8e-2
"""

    return model_directive


def run_pitch_check(
    meshcat,
    record_time,
    pitch_velocity,
    save_time=0.45,
    dt=PITCH_DT,
    model_urdf="iiwa14_limitless",
):
    if meshcat is not None:
        meshcat.Delete()

    builder = DiagramBuilder()

    model_directive = make_model_directive(dt, model_urdf)

    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_context()

    # Just turn off the torque for the time being
    station_context = station.GetMyContextFromRoot(context)
    station.GetInputPort("iiwa.torque").FixValue(station_context, [0] * NUM_JOINTS)

    # Set initial velocity of the ball
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    ball = plant.GetModelInstanceByName("ball")
    plant.SetVelocities(
        plant_context,
        ball,
        np.array([0, 0, 0, pitch_velocity[0], pitch_velocity[1], pitch_velocity[2]]),
    )

    if meshcat is not None:
        meshcat.StartRecording()

    simulator.AdvanceTo(save_time)
    ball_position = plant.GetPositions(plant_context, ball)
    ball_velocity = plant.GetVelocities(plant_context, ball)
    simulator.AdvanceTo(record_time)

    if meshcat is not None:
        meshcat.PublishRecording()

    ball_state = {
        "time": save_time,
        "position": {
            "q0": ball_position[0],
            "q1": ball_position[1],
            "q2": ball_position[2],
            "q3": ball_position[3],
            "x": ball_position[4],
            "y": ball_position[5],
            "z": ball_position[6],
        },
        "velocity": {
            "rx": ball_velocity[0],
            "ry": ball_velocity[1],
            "rz": ball_velocity[2],
            "vx": ball_velocity[3],
            "vy": ball_velocity[4],
            "vz": ball_velocity[5],
        },
    }

    return ball_state
