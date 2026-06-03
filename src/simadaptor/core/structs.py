from flax.struct import dataclass
import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Any, Tuple, Optional
from mujoco import mjx

def print_callback(*args):
    print("Callback:", *args)

@dataclass
class NormStats:
    mean_q: jnp.ndarray
    var_q: jnp.ndarray
    mean_dq: jnp.ndarray
    var_dq: jnp.ndarray
    mean_u: jnp.ndarray
    var_u: jnp.ndarray

@dataclass
class ActuatorParams:
    deadzone: Optional[jnp.ndarray] = None
    torque_bias: Optional[jnp.ndarray] = None
    damping: Optional[jnp.ndarray] = None
    friction_params: Optional[jnp.ndarray] = None  # (..., ndof, 6): [A-, A+, k-, k+, s-, s+]
    torque_scale: Optional[jnp.ndarray] = None  # (..., ndof, 2): [scale_neg, scale_pos]

    def __getitem__(self, idx)->"ActuatorParams":
        """Convenient indexing for dataclass"""
        return jax.tree_util.tree_map(lambda x: x[idx], self) 

@dataclass
class ControllerParams:
    kp: jnp.ndarray
    kd: jnp.ndarray
    torque_range: Optional[jnp.ndarray] = None
    joint_range: Optional[jnp.ndarray] = None
    
    # noise add inside controller
    torque_noise_type: Optional[jnp.ndarray] = None  # e.g., 0: none, 1: white, 2: filtered
    torque_noise_std: Optional[jnp.ndarray] = None  # shape (..., 2, DoF) for white/walk noise


    def __getitem__(self, idx)->"ControllerParams":
        """Convenient indexing for dataclass"""
        return jax.tree_util.tree_map(lambda x: x[idx], self) 
    
    def get_actuator_fn(self, adaptor_model:Optional[nn.Module]=None, adaptor_params=None,
                        history_emb:Optional[jnp.ndarray]=None,
                        ideal_mjx_model:mjx.Model=None,
                        add_noise:bool=False, control_type='qref',
                        enforce_joint_range: bool = True) -> Any:
        def actuator_fn(q_window, qd_window, q_ref, qd_ref, rng, carry=None, u_ref=None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Any]:
            """
            Args:
                q_window/qd_window: [..., T, DoF] observation windows.
                q_ref_window/qd_ref_window: [..., DoF] reference windows (optional if u_ref provided).
                u_ref: [..., T, DoF] window of reference torques (used instead of PD).
            """
            q_last = q_window[..., -1, :]
            qd_last = qd_window[..., -1, :]
            ndof = q_last.shape[-1]

            if control_type == 'qref':
                assert q_ref is not None and qd_ref is not None, "Reference windows required for PD control."
                err = q_ref - q_window[..., -1, :]
                derr = qd_ref - qd_window[..., -1, :]
                tau_ctl = self.kp * err + self.kd * derr

                if enforce_joint_range and self.joint_range is not None:
                    q_min = self.joint_range[...,:ndof,0]
                    q_max = self.joint_range[...,:ndof,1]
                    err_bound = jnp.where(q_window[..., -1, :] < q_min, q_min - q_window[..., -1, :],
                                                 jnp.where(q_window[..., -1, :] > q_max, q_max - q_window[..., -1, :], 0.0))
                    joint_pull_profile = jnp.array([120, 100, 100, 70, 35, 23, 12], dtype=tau_ctl.dtype)
                    joint_pull_idx = jnp.minimum(
                        jnp.arange(ndof, dtype=jnp.int32),
                        jnp.asarray(joint_pull_profile.shape[0] - 1, dtype=jnp.int32),
                    )
                    # joint_pull = 0.1 * joint_pull_profile[joint_pull_idx]
                    # joint_pull = joint_pull_profile[joint_pull_idx]
                    joint_pull = self.kp * 20.0
                    # jax.debug.callback(print_callback, (q_min, q_max, err_bound, joint_pull, q_window[..., -1, :]))
                    tau_ctl = tau_ctl + joint_pull * err_bound
                
                # if velocity is too high, add damping
                vel_threshold = 4.0
                high_vel_mask = jnp.abs(qd_last) > vel_threshold
                vel_damp_profile = jnp.array([10, 10, 10, 10, 4, 4, 4], dtype=tau_ctl.dtype)
                vel_damp_idx = jnp.minimum(
                    jnp.arange(ndof, dtype=jnp.int32),
                    jnp.asarray(vel_damp_profile.shape[0] - 1, dtype=jnp.int32),
                )
                # vel_damp = vel_damp_profile[vel_damp_idx]
                vel_damp = self.kd * 10.0
                tau_ctl = tau_ctl - high_vel_mask * jnp.sign(qd_last) * vel_damp * (jnp.abs(qd_last) - vel_threshold)
                
                assert ideal_mjx_model is not None, "Ideal mjx model must be provided for gravity compensation."
                ideal_mjx_model_ = ideal_mjx_model.replace(body_gravcomp=jnp.zeros_like(ideal_mjx_model.body_gravcomp))
                common_nq = min(q_last.shape[-1], ideal_mjx_model_.nq)
                common_nu = min(tau_ctl.shape[-1], ideal_mjx_model_.nu)
                data0 = mjx.make_data(ideal_mjx_model_)

                def _grav_one(qi: jax.Array) -> jax.Array:
                    qpos = jnp.zeros((ideal_mjx_model_.nq,), dtype=qi.dtype).at[:common_nq].set(qi[:common_nq])
                    qvel = jnp.zeros((ideal_mjx_model_.nv,), dtype=qi.dtype)
                    ctrl = jnp.zeros((ideal_mjx_model_.nu,), dtype=qi.dtype)
                    d = mjx.forward(ideal_mjx_model_, data0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
                    return (d.qfrc_bias - d.qfrc_gravcomp)[:common_nu]

                flat_q = q_last.reshape((-1, q_last.shape[-1]))
                tau_grav = jax.vmap(_grav_one)(flat_q).reshape(q_last.shape[:-1] + (common_nu,))
                tau_ctl = tau_ctl + tau_grav
            
            elif control_type == 'udef':
                tau_ctl_window = u_ref
                tau_ctl = tau_ctl_window[..., -1, :]

            if adaptor_model is not None and adaptor_params is not None:
                # pass through adaptor model with windowed inputs
                q_window, qd_window, tau_ctl_window = jax.tree.map(lambda x: jax.lax.stop_gradient(x), (q_window, qd_window, tau_ctl_window))
                delta_tau, _ = adaptor_model.apply(adaptor_params, q_window, qd_window, tau_ctl_window, history_emb, None)
                assert delta_tau.shape == tau_ctl.shape, f"Adaptor delta_tau shape {delta_tau.shape} mismatch with tau_ctl {tau_ctl.shape}"
                tau_ctl = tau_ctl + delta_tau
            else:
                delta_tau = None
            
            if add_noise:
                rng, noise_key = jax.random.split(rng)
                # Require explicit noise stds when noise is enabled
                if self.torque_noise_std is None:
                    raise ValueError("torque_noise_std must be provided when add_noise is True.")
                noise_std = self.torque_noise_std
                if noise_std.ndim == 2:  # allow (2, DoF) broadcast
                    noise_std = noise_std[None, ...]
                if noise_std.shape[-2] != 2:
                    raise ValueError(f"Expected torque_noise_std last-2 dim=2 (white/walk); got {noise_std.shape}.")
                # Add noise to the torque
                if self.torque_noise_type is not None:
                    tau_noise0 = jnp.zeros_like(tau_ctl)
                    tau_noise1 = jax.random.normal(noise_key, tau_ctl.shape) * noise_std[..., 0, :]
                    
                    tau_noise_prev = jnp.zeros_like(tau_ctl) if carry is None else carry
                    incr = jax.random.normal(noise_key, tau_ctl.shape) * noise_std[..., 1, :]
                    tau_noise2 = tau_noise_prev + incr
                    
                    tau_noise = jnp.where((self.torque_noise_type == 0)[...,None], tau_noise0,
                                         jnp.where((self.torque_noise_type == 1)[...,None], tau_noise1,
                                                   tau_noise2))
                    tau_ctl = tau_ctl + tau_noise
                    carry = tau_noise
            
            return tau_ctl, delta_tau, carry
        return actuator_fn
    
@dataclass
class RolloutParams:
    
    # controller parameters
    kp: jnp.ndarray
    kd: jnp.ndarray
    torque_range: Optional[jnp.ndarray] = None
    joint_range: Optional[jnp.ndarray] = None
    
    # actuator params
    torque_bias: Optional[jnp.ndarray] = None # (..., ndof, 2)
    deadzone: Optional[jnp.ndarray] = None # (..., ndof, 2)
    damping: Optional[jnp.ndarray] = None # (..., ndof, 2)
    friction_params: Optional[jnp.ndarray] = None  # (..., ndof, 6)
    torque_scale: Optional[jnp.ndarray] = None  # (..., ndof, 2): [scale_neg, scale_pos]
    
    # mjx params
    dof_frictionloss: Optional[jnp.ndarray] = None
    dof_armature: Optional[jnp.ndarray] = None # (..., ndof)
    dof_damping: Optional[jnp.ndarray] = None # (..., ndof)
    body_mass: Optional[jnp.ndarray] = None
    body_mass_delta: Optional[jnp.ndarray] = None
    body_inertia: Optional[jnp.ndarray] = None
    body_ipos: Optional[jnp.ndarray] = None
    
    q_noise_std: Optional[jnp.ndarray] = None
    dq_noise_std: Optional[jnp.ndarray] = None
    
    torque_noise_type: Optional[jnp.ndarray] = None  # e.g., 0: none, 1: white, 2: filtered
    torque_noise_std: Optional[jnp.ndarray] = None  # per-noise-type stds for white/walk

    def __getitem__(self, idx)->"RolloutParams":
        """Convenient indexing for dataclass"""
        return jax.tree_util.tree_map(lambda x: x[idx], self) 
    
    def set_mjx_model(self, model: mjx.Model) -> mjx.Model:
        """Apply the rollout parameters to a given mjx.Model."""
        # set default params
        model = model.tree_replace({
                "actuator_ctrlrange": jnp.zeros_like(model.actuator_ctrlrange).at[...,0].set(-jnp.inf).at[...,1].set(jnp.inf),
                "actuator_forcerange": jnp.zeros_like(model.actuator_forcerange).at[...,0].set(-jnp.inf).at[...,1].set(jnp.inf),
                "actuator_actrange": jnp.zeros_like(model.actuator_actrange).at[..., 0].set(-jnp.inf).at[..., 1].set(jnp.inf),
                "jnt_actfrcrange": jnp.zeros_like(model.jnt_actfrcrange).at[...,0].set(-jnp.inf).at[...,1].set(jnp.inf),
                "actuator_gear": jnp.zeros_like(model.actuator_gear).at[...,0].set(1.0),
                "jnt_range": jnp.zeros_like(model.jnt_range).at[...,0].set(-jnp.inf).at[...,1].set(jnp.inf), # remove joint limits
                })
        model = model.replace(dof_frictionloss=jnp.zeros_like(model.dof_frictionloss)) # turn off default friction -> actuator model
        
        # if self.dof_frictionloss is not None:
        #     model = model.tree_replace({"dof_frictionloss": self.dof_frictionloss})
        if self.dof_armature is not None:
            model = model.tree_replace({"dof_armature": self.dof_armature})
        if self.dof_damping is not None:
            model = model.tree_replace({"dof_damping": self.dof_damping})
        body_mass = self.body_mass
        body_inertia = self.body_inertia
        if body_mass is None and self.body_mass_delta is not None:
            base_mass = jnp.asarray(model.body_mass, dtype=jnp.float32)
            body_mass = base_mass + self.body_mass_delta
            positive_body_mask = base_mass > 0
            body_mass = jnp.where(
                positive_body_mask,
                jnp.maximum(body_mass, 1e-2),
                body_mass,
            )
            if body_inertia is None:
                safe_mass = jnp.maximum(base_mass, 1e-8)
                inertia_scale = jnp.where(base_mass > 0, body_mass / safe_mass, 1.0)
                body_inertia = model.body_inertia * inertia_scale[:, None]
        if body_mass is not None:
            model = model.tree_replace({"body_mass": body_mass})
        if body_inertia is not None:
            model = model.tree_replace({"body_inertia": body_inertia})
        if self.body_ipos is not None:
            model = model.tree_replace({"body_ipos": self.body_ipos})
        
        return model

    def from_mjx_model(self, model: mjx.Model) -> "RolloutParams":
        """Initialize the rollout parameters from a given mjx.Model."""
        return RolloutParams(
            kp=None,
            kd=None,
            torque_range=None,
            joint_range=None,
            torque_bias=None,
            deadzone=None,
            damping=None,
            friction_params=None,
            torque_scale=None,
            dof_frictionloss=None,
            dof_armature=model.dof_armature,
            dof_damping=model.dof_damping,
            body_mass=model.body_mass,
            body_mass_delta=None,
            body_inertia=model.body_inertia,
            body_ipos=model.body_ipos,
            q_noise_std=None,
            dq_noise_std=None,
            torque_noise_type=None,
            torque_noise_std=None,
        )
    
    def fit_model_size(self, mjx_model:mjx.Model) -> "RolloutParams":
        """Fit the rollout parameters to the given mjx model size (e.g., when model has extra DoFs).
        Assume articulated robot arm indices are aligned at the front.
        """
        common_nq = min(mjx_model.nq, 7)
        common_nv = min(mjx_model.nv, 7)
        common_nu = min(mjx_model.nu, 7)
        common_nbody = min(mjx_model.nbody, mjx_model.nbody)  # just in case
        
        def _fit_param(param:jnp.ndarray, size:int, axis=-1) -> jnp.ndarray:
            if param is None:
                return None
            else:
                return jnp.moveaxis(jnp.moveaxis(param, axis, -1)[...,:size], -1, axis)
        
        return RolloutParams(
            kp=_fit_param(self.kp, common_nu),
            kd=_fit_param(self.kd, common_nu),
            torque_range=_fit_param(self.torque_range, common_nu, -2),
            joint_range=_fit_param(self.joint_range, common_nu, -2),
            torque_bias=_fit_param(self.torque_bias, common_nu, -2),
            deadzone=_fit_param(self.deadzone, common_nu, -2),
            damping=_fit_param(self.damping, common_nu, -2),
            friction_params=_fit_param(self.friction_params, common_nu, -2),
            torque_scale=_fit_param(self.torque_scale, common_nu, -2),
            dof_frictionloss=_fit_param(self.dof_frictionloss, common_nv),
            dof_armature=_fit_param(self.dof_armature, common_nv),
            dof_damping=_fit_param(self.dof_damping, common_nv),
            body_mass=_fit_param(self.body_mass, common_nbody),
            body_mass_delta=_fit_param(self.body_mass_delta, common_nbody),
            body_inertia=_fit_param(self.body_inertia, common_nbody, -2),
            body_ipos=_fit_param(self.body_ipos, common_nbody, -2),
            q_noise_std=_fit_param(self.q_noise_std, common_nq),
            dq_noise_std=_fit_param(self.dq_noise_std, common_nv),
            torque_noise_type=self.torque_noise_type,
            torque_noise_std=_fit_param(self.torque_noise_std, common_nu, -1) if self.torque_noise_std is not None else None,
        )

    
    @property
    def actuator_params(self) -> ActuatorParams:
        params = ActuatorParams(
            deadzone=self.deadzone,
            torque_bias=self.torque_bias,
            damping=self.damping,
            friction_params=self.friction_params,
            torque_scale=self.torque_scale,
        )
        return params
    
    @property
    def controller_params(self) -> ControllerParams:
        params = ControllerParams(
            kp=self.kp,
            kd=self.kd,
            torque_range=self.torque_range,
            joint_range=self.joint_range,
            torque_noise_type=self.torque_noise_type,
            torque_noise_std=self.torque_noise_std,
        )
        return params
