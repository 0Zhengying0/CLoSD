import bpy
from mathutils import Vector
import os
import sys
sys.path.append('.')

from closd.utils.closd_util import STATES
import closd.blender.blender_utils as blender_utils

import argparse
import numpy as np

object_path_per_state = {
    STATES.SIT: 'closd/data/assets/fbx/sofa.fbx',
    STATES.REACH: 'closd/data/assets/fbx/reach_target.fbx',
    STATES.STRIKE_KICK: 'closd/data/assets/fbx/strike_target.fbx',
    STATES.STRIKE_PUNCH: 'closd/data/assets/fbx/strike_target.fbx',
}


def ensure_headless_render_stage():
    """Complete the repository's empty.blend when rendering without the GUI."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    if bpy.data.materials.get('char') is None:
        material = bpy.data.materials.new('char')
        material.diffuse_color = (0.22, 0.65, 0.16, 1)
    if bpy.data.materials.get('marker') is None:
        material = bpy.data.materials.new('marker')
        material.diffuse_color = (0.08, 0.55, 0.9, 1)
    if scene.camera is None:
        camera_data = bpy.data.cameras.new('Camera')
        camera = bpy.data.objects.new('Camera', camera_data)
        scene.collection.objects.link(camera)
        camera.location = (4, -6, 3.5)
        direction = Vector((0, 0, 1)) - camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        camera_data.type = 'ORTHO'
        camera_data.ortho_scale = 8
        scene.camera = camera
    if bpy.data.objects.get('CheckerFloor') is None:
        bpy.ops.mesh.primitive_plane_add(size=200)
        floor = bpy.context.object
        floor.name = 'CheckerFloor'
        floor_mat = bpy.data.materials.new('CheckerFloor')
        floor_mat.use_nodes = True
        nodes = floor_mat.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        diffuse = nodes.new('ShaderNodeBsdfPrincipled')
        checker = nodes.new('ShaderNodeTexChecker')
        checker.inputs['Color1'].default_value = (0.75, 0.75, 0.75, 1)
        checker.inputs['Color2'].default_value = (0.22, 0.22, 0.22, 1)
        checker.inputs['Scale'].default_value = 240
        floor_mat.node_tree.links.new(checker.outputs['Color'], diffuse.inputs['Base Color'])
        floor_mat.node_tree.links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
        floor.data.materials.append(floor_mat)
    if bpy.data.objects.get('QualityAreaLight') is None:
        light_data = bpy.data.lights.new('QualityAreaLight', 'AREA')
        light = bpy.data.objects.new('QualityAreaLight', light_data)
        scene.collection.objects.link(light)
        light.location = (0, -3, 6)
        light_data.energy = 900
        light_data.shape = 'DISK'
        light_data.size = 6
    scene.world.color = (0.25, 0.25, 0.25)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, required=True, help='Input pkl file recording states')
    parser.add_argument("--results_dir", type=str, default='closd/blender/results', help='')
    parser.add_argument("--stage_path", type=str, default='closd/blender/assets/empty.blend', help='Empty blender file (possibly with predefined camera and lights).')
    parser.add_argument("--mjcf_path", type=str, default='closd/data/assets/mjcf/smpl_humanoid.xml', help='mjcf file defining the humanoid.')
    parser.add_argument("--cam_track", action='store_true', help='if true, will enable camera tracking.')
    parser.add_argument("--add_markers", action='store_true', help='if true, will add MDM moption markers.')
    parser.add_argument("--full_episods", action='store_true', help='if true, will render only full episods.')
    parser.add_argument("--single_file", action='store_true', help='if true, will put all motions in a single file.')
    parser.add_argument("--downsample_rate", type=int, default=2, help='')
    parser.add_argument("--prepared_npz", type=str, default=None,
                        help='Precomputed real simulator episode; bypasses torch imports in Blender Python.')
    parser.add_argument("--assets_dir", type=str, default='closd/blender/assets',
                        help='Directory containing official xml2mesh humanoid OBJs.')
    parser.add_argument("--render_mp4", type=str, default=None,
                        help='Also render the prepared single episode directly to MP4.')
    parser.add_argument("--preview_frames", type=int, default=0,
                        help='If positive, render only this many frames; for technical smoke tests.')
    
    if "--" not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index("--") + 1:]
    params = parser.parse_args(argv)

    orig_framerate = 60  # fps
    new_framerate = int(orig_framerate / params.downsample_rate)
    if params.prepared_npz:
        with np.load(params.prepared_npz, allow_pickle=False) as prepared:
            sample_name = str(prepared['sample_name'].item())
            record_data = {sample_name: {key: prepared[key].copy() for key in prepared.files}}
        if params.render_mp4 and params.single_file:
            raise ValueError('render_mp4 requires a single episode file')
    else:
        import joblib
        record_data = joblib.load(params.record_path)
    all_n_frames = [(len(data['pos_data']) if params.prepared_npz else
                     len(data['ref_body_pos_full'][0::params.downsample_rate]))
                    for data in record_data.values()]
    full_episod_len = max(all_n_frames)
    print(f'Full episod length is [{full_episod_len}]')

    all_path = os.path.join(params.results_dir, os.path.basename(params.record_path).replace('.pkl', ''), 'all.blend')
    if params.single_file:
        bpy.ops.wm.open_mainfile(filepath=params.stage_path)

    for sample_i, (sample_name, sample_data) in enumerate(record_data.items()):

        env_id, try_id = [int(s) for s in sample_name.split('_')]
        out_path = os.path.join(params.results_dir, os.path.basename(params.record_path).replace('.pkl', ''), sample_name + '.blend')
        markers_data = (sample_data['markers_data'] if params.prepared_npz else
                        sample_data['ref_body_pos_full'][0::params.downsample_rate])
        n_frames, n_joints, _ = markers_data.shape
        
        if params.full_episods and n_frames < full_episod_len:
            print(f'Skipping [{sample_name}]')
            continue

        if params.prepared_npz:
            pos_data = sample_data['pos_data'].copy()
            euler_data = sample_data['euler_data'].copy()
        else:
            import torch
            from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
            pose_quat, trans = sample_data['body_quat'].numpy()[0::params.downsample_rate], sample_data['trans'].numpy()[0::params.downsample_rate]
            skeleton_tree = SkeletonTree.from_dict(sample_data['skeleton_tree'])
            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=True)
            quat_data = sk_state.global_rotation
            pos_data = sk_state.global_translation.cpu().numpy()
            euler_data = blender_utils.quaternion_to_euler_angle_vectorized1(quat_data.reshape(n_frames*n_joints, 4)).reshape(n_frames, n_joints, 3)
        
        # smooth_end_effectors:
        euler_data = blender_utils.smooth_end_effectors(euler_data, window_size=10)
        pos_data = blender_utils.smooth_end_effectors(pos_data)


        print(f'Visualizing sample [{sample_name}] with [{n_frames}] frames.')
        if not params.single_file:
            bpy.ops.wm.open_mainfile(filepath=params.stage_path)
        if params.render_mp4:
            ensure_headless_render_stage()
        blender_utils.load_humanoid(params.mjcf_path, assets_dir=params.assets_dir, motion_name=sample_name)
        if params.add_markers:
            blender_utils.add_full_body_markers(sample_name)
        scn = bpy.context.scene
        scn.frame_start = 1
        scn.frame_end = n_frames
        scn.render.fps = new_framerate
        for joint_i, joint_name in enumerate(blender_utils.mujoco_joint_names):
            _obj = blender_utils.get_object_by_idx(sample_name, joint_i)
            for frame_i in range(n_frames):
                # Set location-rotation as a keyframe
                _pos = pos_data[frame_i, joint_i]
                _rot = euler_data[frame_i, joint_i]
                _obj.location = Vector(_pos)
                _obj.rotation_euler = Vector(_rot)
                _obj.keyframe_insert('location', frame=frame_i+1)
                _obj.keyframe_insert('rotation_euler', frame=frame_i+1)
                if params.add_markers:
                    _marker = bpy.data.objects[sample_name + '_marker_' + joint_name]
                    _marker_pos = markers_data[frame_i, joint_i]
                    _marker.location = Vector(_marker_pos)
                    _marker.keyframe_insert('location', frame=frame_i+1)
        
        # add target object
        has_target = ('target_trans' in sample_data.keys() and 'init_state' in sample_data.keys()
                      and np.asarray(sample_data['target_trans']).size > 0)
        if has_target:
            target_coll = blender_utils.create_collection(sample_name + "_target")
            default_collection = bpy.data.collections["Collection"]
            
            if len(sample_data['target_trans'].shape) == 2:
                init_state = sample_data['init_state']
                init_state = int(init_state.cpu().numpy()) if hasattr(init_state, 'cpu') else int(init_state)
                target_path = object_path_per_state.get(init_state)
                if target_path is None:
                    raise ValueError('no target FBX for initial task {}'.format(init_state))
                bpy.ops.import_scene.fbx(filepath=target_path)
                if params.prepared_npz:
                    target_trans = sample_data['target_trans']
                    target_euler = sample_data['target_euler'].copy()
                else:
                    target_trans = sample_data['target_trans'].cpu().numpy()[0::params.downsample_rate]
                    target_quat = sample_data['target_quat'].cpu().numpy()[0::params.downsample_rate]
                    target_euler = blender_utils.quaternion_to_euler_angle_vectorized1(target_quat)
                if 'sofa' in target_path:
                    target_euler[:, -1] -= np.pi/2
                target_obj = bpy.data.objects['target']
                target_obj.name = sample_name + '_' + target_obj.name
                blender_utils.move_collection(target_obj, default_collection, target_coll)
                for frame_i in range(1, n_frames):
                    target_obj.location = Vector(target_trans[frame_i])
                    target_obj.rotation_euler = Vector(target_euler[frame_i])
                    target_obj.keyframe_insert('location', frame=frame_i+1)
                    target_obj.keyframe_insert('rotation_euler', frame=frame_i+1)
            if len(sample_data['target_trans'].shape) == 3:
                target_path = [object_path_per_state[int(STATES.REACH)], object_path_per_state[int(STATES.STRIKE_PUNCH)], object_path_per_state[int(STATES.SIT)]]
                obj_names = ['reach_target', 'strike_target', 'sit_target']
                for p, name in zip(target_path, obj_names):
                    bpy.ops.import_scene.fbx(filepath=p)
                    bpy.data.objects['target'].name = name
                if params.prepared_npz:
                    target_trans = sample_data['target_trans']
                    target_euler = sample_data['target_euler'].copy()
                else:
                    target_trans = sample_data['target_trans'].cpu().numpy()[0::params.downsample_rate]
                    target_quat = sample_data['target_quat'].cpu().numpy()[0::params.downsample_rate]
                    target_euler = blender_utils.quaternion_to_euler_angle_vectorized1(target_quat)
                target_euler[..., -1, -1] -= np.pi/2  # sofa case
                for obj_i, name in enumerate(obj_names):
                    target_obj = bpy.data.objects[name]
                    target_obj.name = sample_name + '_' + target_obj.name
                    blender_utils.move_collection(target_obj, default_collection, target_coll)
                    for frame_i in range(1, n_frames):
                        target_obj.location = Vector(target_trans[frame_i, obj_i])
                        target_obj.rotation_euler = Vector(target_euler[frame_i, obj_i])
                        target_obj.keyframe_insert('location', frame=frame_i+1)
                        target_obj.keyframe_insert('rotation_euler', frame=frame_i+1)

        
        # Camera tracking
        if params.cam_track:
            camera = bpy.data.objects['Camera']
            for frame_i in range(1, n_frames):
                delta = pos_data[frame_i, 0] - pos_data[frame_i-1, 0]
                camera.location.x += delta[0]
                camera.location.y += delta[1]
                # camera.location.z += delta[2]
                camera.keyframe_insert('location', frame=frame_i+1)
        
        # Saving result
        if not params.single_file:
            if not params.render_mp4:
                blender_utils.remove_collection("Collection")
            blender_utils.save_blend(out_path)
            if params.render_mp4:
                scn.frame_end = min(n_frames, params.preview_frames) if params.preview_frames else n_frames
                scn.render.image_settings.file_format = 'FFMPEG'
                scn.render.ffmpeg.format = 'MPEG4'
                scn.render.ffmpeg.codec = 'H264'
                scn.render.filepath = os.path.abspath(params.render_mp4)
                bpy.ops.render.render(animation=True)
                if not os.path.isfile(params.render_mp4) or os.path.getsize(params.render_mp4) < 1024:
                    raise RuntimeError('Blender MP4 was not produced')
    
    if params.single_file:
        blender_utils.remove_collection("Collection")
        blender_utils.save_blend(all_path)
