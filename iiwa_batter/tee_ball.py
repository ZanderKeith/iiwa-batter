import numpy as np

from pydrake.all import (
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RollPitchYaw,
    Simulator,
)

from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.utils import ConfigureParser

from iiwa_batter import PACKAGE_ROOT

# TODO: just had a brainwave
# I don't need to worry about locking joints into a given position if I change the urdf to do that for me

tee_x = -0.05
tee_y = -1.15

model_directive = f"""
directives: 
- add_model:
    name: iiwa
    file: file://{PACKAGE_ROOT}/assets/iiwa14_tee_ball.urdf
    default_joint_positions:
        iiwa_joint_1: [1.57]
        iiwa_joint_4: [0]
        iiwa_joint_7: [-3]
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

def run_tee_ball(meshcat, record_time=3.0):
    meshcat.Delete()

    builder = DiagramBuilder()
    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)

    context = simulator.get_context()
    station_context = station.GetMyContextFromRoot(context)

    #driving_torque = np.array([320, -100, 0, 0, 0])
    # Torque limits are 320, 176, 40
    driving_torque = np.array([0, 0, 40])
    station.GetInputPort("iiwa.torque").FixValue(station_context, driving_torque)


    meshcat.StartRecording()
    simulator.AdvanceTo(record_time)
    meshcat.PublishRecording()