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

model_directive = f"""
directives: 
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
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
            translation: [0.5, 0, 0.5]
- add_model:
    name: tee
    file: file://{PACKAGE_ROOT}/assets/tee.sdf
- add_weld:
    parent: world
    child: tee::base
    X_PC:
        translation: [0.5, 0, -0.2]
"""

def run_tee_ball(meshcat, record_time=3.0):
    meshcat.Delete()
    scenario = LoadScenario(data=model_directive)
    station = MakeHardwareStation(
        scenario, meshcat
    )
    simulator = Simulator(station)
    meshcat.StartRecording()
    simulator.AdvanceTo(record_time)
    meshcat.PublishRecording()