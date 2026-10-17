import math

import torch


class VelocityODESampler:
    """
    Shared ODE sampler for velocity-prediction generative models.

    Intended for both:
      - Flow Matching
      - Rectified Flow

    Convention used by our training:
        t = 0  -> Gaussian noise
        t = 1  -> motion data

    ODE:
        dx/dt = v_theta(x, t, condition)

    The model still receives integer timestep indices because the
    current MDM uses a discrete positional-encoding lookup.
    """

    def __init__(
        self,
        num_timesteps=1000,
        num_steps=10,
        solver="euler",
        rtol=1e-3,
        atol=1e-4,
        max_num_steps=128,
        max_nfe=256,
    ):
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be >= 2")

        if num_steps < (0 if solver == "dopri5" else 1):
            raise ValueError("num_steps must be >= 0 for DOPRI5 or >= 1 for fixed solvers")

        if solver not in ["euler", "heun", "rk4", "dopri5"]:
            raise ValueError(
                f"Unsupported solver [{solver}]. "
                "Choose from ['euler', 'heun', 'rk4', 'dopri5']."
            )
        if solver == "dopri5" and (
            not math.isfinite(rtol) or not math.isfinite(atol)
            or rtol <= 0 or atol <= 0
            or max_num_steps < 1 or max_nfe < 1
        ):
            raise ValueError("DOPRI5 requires positive finite tolerances and limits")

        self.num_timesteps = num_timesteps
        self.num_steps = num_steps
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_num_steps = max_num_steps
        self.max_nfe = max_nfe
        self.last_nfe = 0

    @property
    def nfe(self):
        """
        Number of network function evaluations.

        Euler:
            1 model call per integration step.

        Heun:
            2 model calls per integration step.

        RK4:
            4 model calls per integration step.

        DOPRI5:
            Measured velocity evaluations from the most recent solve.

        This counts velocity evaluations, not the underlying MDM forwards.
        ClassifierFreeSampleModel runs the MDM twice per evaluation.
        """
        if self.solver == "euler":
            return self.num_steps

        if self.solver == "heun":
            return 2 * self.num_steps

        if self.solver == "rk4":
            return 4 * self.num_steps

        if self.solver == "dopri5":
            return self.last_nfe

        raise ValueError(self.solver)

    def _time_to_index(self, tau, batch_size, device):
        """
        Convert continuous tau in [0, 1] into the integer timestep
        expected by the current MDM timestep embedding.

        Example for T=1000:
            tau=0   -> 0
            tau=1   -> 999
        """
        tau = float(tau)

        idx = int(round(
            tau * (self.num_timesteps - 1)
        ))

        idx = max(
            0,
            min(self.num_timesteps - 1, idx),
        )

        return torch.full(
            (batch_size,),
            idx,
            dtype=torch.long,
            device=device,
        )

    def _model_velocity(
        self,
        model,
        x,
        tau,
        model_kwargs,
    ):
        """
        Evaluate v_theta(x_tau, tau, condition).
        """
        t_idx = self._time_to_index(
            tau,
            batch_size=x.shape[0],
            device=x.device,
        )

        # MDM.forward() rebinds y['mask'] after appending the prefix, and
        # may add other keys. Keep each ODE stage's condition dictionary
        # independent without copying its large, read-only tensors.
        call_kwargs = model_kwargs.copy()
        if "y" in call_kwargs and call_kwargs["y"] is not None:
            call_kwargs["y"] = call_kwargs["y"].copy()

        return model(
            x,
            t_idx,
            **call_kwargs,
        )

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        recon_guidance=False,
        dump_steps=None,
        const_noise=False,
        **unused_kwargs,
    ):
        """
        Integrate the learned velocity field from t=0 to t=1.

        This function intentionally mimics the interface of
        GaussianDiffusion.p_sample_loop(), so existing CLoSD sampling
        code can call it with minimal changes.

        Returns:
            Generated future motion.

        If model_kwargs['y']['prefix'] exists, the prefix is appended
        to the returned motion exactly like the current diffusion
        sampler.
        """

        if model_kwargs is None:
            model_kwargs = {"y": {}}

        # These diffusion-specific features have not yet been ported
        # to the ODE sampler. Fail explicitly rather than silently
        # producing a wrong result.
        if skip_timesteps != 0:
            raise NotImplementedError(
                "skip_timesteps is not supported by VelocityODESampler."
            )

        if init_image is not None:
            raise NotImplementedError(
                "init_image is not supported by VelocityODESampler yet."
            )

        if cond_fn is not None:
            raise NotImplementedError(
                "classifier guidance cond_fn is not supported yet."
            )

        if cond_fn_with_grad:
            raise NotImplementedError(
                "cond_fn_with_grad is not supported yet."
            )

        if recon_guidance:
            raise NotImplementedError(
                "recon_guidance is not supported yet."
            )

        if randomize_class:
            raise NotImplementedError(
                "randomize_class is not supported."
            )

        if denoised_fn is not None:
            raise NotImplementedError(
                "denoised_fn is not supported."
            )

        if device is None:
            if noise is not None:
                device = noise.device
            else:
                device = next(model.parameters()).device

        # ---------------------------------------------------------
        # Initial state x(0): Gaussian noise.
        # ---------------------------------------------------------
        if noise is None:
            if const_noise:
                single_noise = torch.randn(
                    1,
                    *shape[1:],
                    device=device,
                )

                x = single_noise.repeat(
                    shape[0],
                    *([1] * (len(shape) - 1)),
                )
            else:
                x = torch.randn(
                    *shape,
                    device=device,
                )
        else:
            x = noise.to(device)

            if tuple(x.shape) != tuple(shape):
                raise ValueError(
                    f"noise shape {tuple(x.shape)} "
                    f"does not match requested shape {tuple(shape)}"
                )

        if self.solver == "dopri5":
            if dump_steps is not None:
                raise NotImplementedError("dump_steps is not defined for adaptive DOPRI5")
            try:
                from torchdiffeq import odeint
            except ImportError as exc:
                raise ImportError(
                    "DOPRI5 requires torchdiffeq==0.2.5 in the CLoSD Python environment"
                ) from exc

            self.last_nfe = 0

            def velocity(tau, state):
                if self.last_nfe >= self.max_nfe:
                    raise RuntimeError(
                        f"DOPRI5 exceeded max_nfe={self.max_nfe}; "
                        "the rounded MDM time index can cause adaptive step rejections"
                    )
                self.last_nfe += 1
                # _model_velocity uses the same nearest integer MDM index as
                # Euler, Heun and RK4. It copies the mutable y mapping on
                # every call, including rejected adaptive stages.
                return self._model_velocity(model, state, tau, model_kwargs)

            times = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
            x = odeint(
                velocity, x, times, method="dopri5", rtol=self.rtol,
                atol=self.atol, options={"max_num_steps": self.max_num_steps},
            )[-1]
            if not torch.isfinite(x).all():
                raise RuntimeError("DOPRI5 produced NaN or Inf motion")
            if "y" in model_kwargs and "prefix" in model_kwargs["y"]:
                x = torch.cat([model_kwargs["y"]["prefix"], x], dim=-1)
            return x

        # Integration grid:
        #
        #   0 = t_0 < ... < t_N = 1
        #
        time_grid = torch.linspace(
            0.0,
            1.0,
            self.num_steps + 1,
            device=device,
        )

        iterator = range(self.num_steps)

        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(
                iterator,
                desc=f"ODE {self.solver}, NFE={self.nfe}",
            )

        dumped = []

        # ---------------------------------------------------------
        # Solve dx/dt = v_theta(x,t,c)
        # ---------------------------------------------------------
        for step_i in iterator:
            t0 = time_grid[step_i].item()
            t1 = time_grid[step_i + 1].item()

            dt = t1 - t0

            v0 = self._model_velocity(
                model,
                x,
                t0,
                model_kwargs,
            )

            if self.solver == "euler":
                # Euler:
                #
                # x_{i+1} = x_i + dt * v(x_i, t_i)
                x = x + dt * v0

            elif self.solver == "heun":
                # Predictor.
                x_euler = x + dt * v0

                # Velocity at the predicted endpoint.
                v1 = self._model_velocity(
                    model,
                    x_euler,
                    t1,
                    model_kwargs,
                )

                # Corrector.
                x = x + 0.5 * dt * (v0 + v1)

            elif self.solver == "rk4":
                # Classical explicit RK4, always integrating forward 0 -> 1.
                # Every stage uses the same continuous-to-discrete MDM time
                # mapping as Euler and Heun. The quantized velocity field is
                # discontinuous in time, so formal fourth-order convergence
                # is not guaranteed for this model.
                t_mid = t0 + 0.5 * dt
                v1 = self._model_velocity(
                    model, x + 0.5 * dt * v0, t_mid, model_kwargs,
                )
                v2 = self._model_velocity(
                    model, x + 0.5 * dt * v1, t_mid, model_kwargs,
                )
                v3 = self._model_velocity(
                    model, x + dt * v2, t1, model_kwargs,
                )
                x = x + (dt / 6.0) * (v0 + 2.0 * v1 + 2.0 * v2 + v3)

            if dump_steps is not None and step_i in dump_steps:
                dumped.append(x.clone())

        if dump_steps is not None:
            return dumped

        # Match the old diffusion sampler behavior:
        # prefix-completion model returns prefix + generated suffix.
        if (
            "y" in model_kwargs
            and "prefix" in model_kwargs["y"]
        ):
            x = torch.cat(
                [
                    model_kwargs["y"]["prefix"],
                    x,
                ],
                dim=-1,
            )

        return x
