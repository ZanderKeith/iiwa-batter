import numpy as np
import pydot
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

from iiwa_batter import PACKAGE_ROOT, PITCH_DT
from iiwa_batter.physics import exit_velo_mph, parse_ball_state

# TODO: just had a brainwave
# I don't need to worry about locking joints into a given position if I change the urdf to do that for me

tee_x = -0.05
tee_y = -1.15


def make_model_directive(initial_joint_positions):
    model_directive = f"""
directives:
- add_model:
    name: iiwa
    file: file://{PACKAGE_ROOT}/assets/iiwa14_tee_ball.urdf
    default_joint_positions:
        iiwa_joint_1: [{initial_joint_positions[0]}]
        iiwa_joint_4: [{initial_joint_positions[1]}]
        iiwa_joint_7: [{initial_joint_positions[2]}]
- add_weld:
    parent: world
    child: iiwa::base
- add_model:
    name: bat
    file: file://{PACKAGE_ROOT}/assets/bat.sdf
- add_weld:
    parent: iiwa::iiwa_link_ee
    child: bat::base
    X_PC:
        translation: [0, 0, 0.2]
- add_model:
    name: ball
    file: file://{PACKAGE_ROOT}/assets/ball.sdf
    default_free_body_pose:
        base:
            translation: [{tee_x}, {tee_y}, 0.5]
- add_model:
    name: tee
    file: file://{PACKAGE_ROOT}/assets/tee.sdf
- add_weld:
    parent: world
    child: tee::base
    X_PC:
        translation: [{tee_x}, {tee_y}, -0.05]

model_drivers:
    iiwa: !IiwaDriver
      hand_model_name: wsg
      control_mode: torque_only
"""

    return model_directive


def run_tee_ball(
    meshcat,
    joint_positions=None,
    driving_torque=None,
    record_time=3.0,
    dt=PITCH_DT,
):
    if meshcat is not None:
        meshcat.Delete()

    builder = DiagramBuilder()

    model_directive = make_model_directive(joint_positions)

    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)

    context = simulator.get_context()
    station_context = station.GetMyContextFromRoot(context)

    # driving_torque = np.array([320, -100, 0, 0, 0])
    # Torque limits are 320, 176, 40
    driving_torque = np.array(driving_torque)
    station.GetInputPort("iiwa.torque").FixValue(station_context, driving_torque)

    # Record velocity of the ball

    # RenderDiagram(station, max_depth=1)
    pydot.graph_from_dot_data(station.GetGraphvizString(max_depth=2))[0].write_png(
        "station_render.png"
    )

    if meshcat is not None:
        meshcat.StartRecording()
    # Advance simulation for many time steps

    ball_states = []
    joint_velocities = []

    for time in np.linspace(0, record_time, int(record_time / dt)):
        simulator.AdvanceTo(time)
        ball_state = station.GetOutputPort("ball_state").Eval(station_context)
        joint_velocity = station.GetOutputPort("iiwa.velocity_estimated").Eval(
            station_context
        )
        joint_velocities.append(joint_velocity)
        zeroed_torque = driving_torque.copy()
        for i, velocity_component in enumerate(joint_velocity):
            if velocity_component > np.pi * 4:
                zeroed_torque[i] = -10
            elif velocity_component < -np.pi * 4:
                zeroed_torque[i] = 10
        station.GetInputPort("iiwa.torque").FixValue(station_context, zeroed_torque)

        ball_states.append(parse_ball_state(ball_state))

    # print(joint_velocities)

    if meshcat is not None:
        meshcat.PublishRecording()

    return exit_velo_mph(ball_states)
