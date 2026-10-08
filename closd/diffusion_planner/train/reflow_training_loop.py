from closd.diffusion_planner.train.training_loop import TrainLoop


class ReflowTrainLoop(TrainLoop):
    """
    TrainLoop for RF-2.

    Reflow pairs already contain their exact conditioning.
    Therefore conditions must NOT be regenerated during training.
    """

    def cond_modifiers(self, cond, motion):

        if self.args.spatial_condition is not None:
            raise NotImplementedError(
                "RF-2 reflow training currently does not support "
                "spatial_condition."
            )

        if self.args.keyframe_cond_type != "":
            raise NotImplementedError(
                "RF-2 reflow training currently does not support "
                "keyframe conditioning."
            )

        required = [
            "reflow_noise",
            "prefix",
            "target_cond",
            "target_joint_names",
            "is_heading",
        ]

        missing = [
            key for key in required
            if key not in cond
        ]

        if missing:
            raise KeyError(
                f"RF-2 batch missing conditions: {missing}"
            )

        # Deliberately do nothing.
        #
        # In the original TrainLoop:
        #   target_cond_modifier()
        # would randomly regenerate target conditions.
        #
        # That would destroy the z0/z1/condition pairing.
        return
