# Official CLoSD executed-motion renderer

Source: real CLoSD simulator records in execution/dip and execution/fm. Body meshes and keyframe animation use closd/blender/record2anim.py, blender_utils.py, the repository MJCF and FBX assets. These are closed-loop controller motions, distinct from offline planner trajectories. The two models use the same selected episode key and seed; their later trajectories can diverge.
