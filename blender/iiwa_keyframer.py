bl_info = {
    "name": "Iiwa Keyframer",
    "blender": (4, 30, 0),
    "category": "Object",
}

import bpy
from mathutils import Euler

# Dictionary that contains info on how each joint in the armature should rotate
# This converts from Drake rotation to Blender rotation
# (moreso just a conversion to this particular rig because I'm new to rigging and wasn't consistent)
# For this to line up, the rig's transform should be set to location y = 1m, rotation z = 135 degrees

ARMATURE_DICT = {
    "link_1": {
        "index": 0, # Index in the FK vector
        "axis": 1, # Index in the blender keyframe
        "inverted": False, # To flip the rotation direction so it matches Drake
    },
    "link_2": {
        "index": 1,
        "axis": 2,
        "inverted": True,
    },
    "link_3": {
        "index": 2,
        "axis": 1,
        "inverted": False,
    },
    "link_4": {
        "index": 3,
        "axis": 2,
        "inverted": False,
    },
    "link_5": {
        "index": 4,
        "axis": 1,
        "inverted": False,
    },
    "link_6": {
        "index": 5,
        "axis": 1,
        "inverted": False,
    },
    "link_7": {
        "index": 6,
        "axis": 1,
        "inverted": False,
    }
}

D = bpy.data
C = bpy.context



class IiwaKeyframer(bpy.types.Operator):
    """Iiwa Keyframer"""
    bl_idname = "object.iiwa_keyframer"
    bl_label = "Iiwa Keyframer"
    bl_options = {'REGISTER', 'UNDO'}

    text_name: bpy.props.StringProperty(name="Keyframe Text", default="iiwa_keyframes")

    def execute(self, context):
        # Get the armature named iiwa14
        iiwa_armature = context.active_object
        bpy.ops.object.mode_set(mode='POSE')

        if not iiwa_armature:
            self.report({'ERROR'}, "No object selected!")
            return {'CANCELLED'}
    
        if self.text_name in bpy.data.texts:
            text_object = bpy.data.texts[self.text_name]
        else:
            self.report({'ERROR'}, f"Text object {self.text_name} doesn't exist!")
            return {'CANCELLED'}
        
        text_data = text_object.as_string()

        trajectory = []
        lines = text_data.splitlines()
        for i, line in enumerate(lines):
            # First line should be q1,q2,q3,q4,q5,q6,q7
            if i == 0:
                continue
            try:
                position_strings = line.split(',')
                position = []
                for pos in position_strings:
                    position.append(float(pos))
                trajectory.append((i, position))
            except ValueError:
                print(f"Skipping invalid line: {line}")

        for pose in trajectory:
            frame = pose[0]
            joint_angles = pose[1]
            # Iterate through each bone in the armature
            for bone_key in ARMATURE_DICT.keys():
                # Get the bone details to put rotation on the right axis
                bone = iiwa_armature.pose.bones[bone_key]
                bone_details = ARMATURE_DICT[bone_key]
                axis = bone_details["axis"]
                angle = joint_angles[bone_details["index"]]
                if bone_details["inverted"]:
                    angle = angle * -1
                euler_angles = [0, 0, 0]
                euler_angles[axis] = angle
                
                # Insert a keyframe for the bone's quaternion rotation
                quaternion = Euler(euler_angles, 'XYZ').to_quaternion()
                if bone.rotation_quaternion.dot(quaternion) < 0: # make quaternion go in shortest direction
                    quaternion.negate()
                bone.rotation_quaternion = quaternion
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, f"Applied {len(trajectory)} keyframes to {iiwa_armature.name}")
        return {'FINISHED'}

def menu_func(self, context):
    self.layout.operator(IiwaKeyframer.bl_idname)

# store keymaps here to access after registration
addon_keymaps = []

def register():
    bpy.utils.register_class(IiwaKeyframer)
    bpy.types.VIEW3D_MT_object.append(menu_func)

    # handle the keymap
    wm = bpy.context.window_manager
    # Note that in background mode (no GUI available), keyconfigs are not available either,
    # so we have to check this to avoid nasty errors in background case.
    kc = wm.keyconfigs.addon
    if kc:
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
        kmi = km.keymap_items.new(IiwaKeyframer.bl_idname, 'I', 'PRESS', ctrl=True, shift=True)
        kmi.properties.text_name = "iiwa_keyframes"
        addon_keymaps.append((km, kmi))

def unregister():
    # Note: when unregistering, it's usually good practice to do it in reverse order you registered.
    # Can avoid strange issues like keymap still referring to operators already unregistered...
    # handle the keymap
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.utils.unregister_class(IiwaKeyframer)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == "__main__":
    register()