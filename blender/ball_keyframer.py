bl_info = {
    "name": "Ball Keyframer",
    "blender": (4, 30, 0),
    "category": "Object",
}

import bpy

class BallKeyframer(bpy.types.Operator):
    """Ball Keyframer"""
    bl_idname = "object.ball_keyframer"
    bl_label = "Ball Keyframer"
    bl_options = {'REGISTER', 'UNDO'}

    text_name: bpy.props.StringProperty(name="Keyframe Text", default="ball_keyframes")

    def execute(self, context):
        active_object = context.active_object

        if not active_object:
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
            # First line should be x,y,z
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
            position = pose[1]
            active_object.location = pose[1]
            active_object.keyframe_insert(data_path="location", frame=frame)

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, f"Applied {len(trajectory)} keyframes")
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(BallKeyframer.bl_idname)

# store keymaps here to access after registration
addon_keymaps = []

def register():
    bpy.utils.register_class(BallKeyframer)
    bpy.types.VIEW3D_MT_object.append(menu_func)

    # handle the keymap
    wm = bpy.context.window_manager
    # Note that in background mode (no GUI available), keyconfigs are not available either,
    # so we have to check this to avoid nasty errors in background case.
    kc = wm.keyconfigs.addon
    if kc:
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
        kmi = km.keymap_items.new(BallKeyframer.bl_idname, 'B', 'PRESS', ctrl=True, shift=True)
        kmi.properties.text_name = "ball_keyframes"
        addon_keymaps.append((km, kmi))

def unregister():
    # Note: when unregistering, it's usually good practice to do it in reverse order you registered.
    # Can avoid strange issues like keymap still referring to operators already unregistered...
    # handle the keymap
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.utils.unregister_class(BallKeyframer)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == "__main__":
    register()