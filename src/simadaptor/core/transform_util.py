from __future__ import annotations

"""
Shared rigid-transform utilities.

This is the canonical repo-wide home for generic SE(3), rotation-matrix, and
quaternion math used by TAM runtime, simulation, and tests.
"""
import jax.numpy as jnp
import numpy as np
import jax
import einops


def _is_jax_array(x) -> bool:
    return type(x).__module__.startswith(("jax", "jaxlib"))


def _xp(x):
    # pick numpy or jax.numpy based on array type
    return jnp if _is_jax_array(x) else np

def rand_sphere(outer_shape):
    ext = np.random.normal(size=outer_shape + (5,))
    return (ext / np.linalg.norm(ext, axis=-1, keepdims=True))[...,-3:]

def safe_norm(x, axis, keepdims=False, eps=0.0):
    xp = _xp(x)
    is_zero = xp.all(xp.isclose(x,0.), axis=axis, keepdims=True)
    # temporarily swap x with ones if is_zero, then swap back
    x = xp.where(is_zero, xp.ones_like(x), x)
    n = xp.linalg.norm(x, axis=axis, keepdims=keepdims)
    n = xp.where(is_zero if keepdims else xp.squeeze(is_zero, -1), 0., n)
    return n.clip(eps)

# def safe_norm(v, axis=-1, keepdims=False, eps=1e-15):
#     return jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=axis, keepdims=keepdims), eps))

def normalize(v, axis=-1, eps=1e-15):
    xp = _xp(v)
    n = xp.linalg.norm(v, axis=axis, keepdims=True)
    inv = xp.where(n > eps, 1.0 / n, 0.0)
    return xp.where(n > eps, v * inv, v)


def _canonical_quat(order: str):
    name = str(order).strip().lower()
    if name not in {"xyzw", "wxyz"}:
        raise ValueError(f"Unsupported quaternion order '{order}'. Choose from: xyzw, wxyz.")
    if name == "xyzw":
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _normalize_quat_impl(q, *, order: str, eps: float = 1e-8, zero_policy: str = "raise"):
    xp = _xp(q)
    arr = xp.asarray(q, dtype=xp.float32 if xp is jnp else np.float32)
    n = xp.linalg.norm(arr, axis=-1, keepdims=True)
    safe_n = xp.maximum(n, eps)
    normalized = arr / safe_n
    if zero_policy == "raise":
        if xp is np and np.any(n <= eps):
            raise ValueError("Quaternion norm must be positive.")
        # JAX cannot eagerly raise for traced values; keep a stable identity fallback.
        zero_policy = "identity"
    if zero_policy == "identity":
        default = xp.asarray(_canonical_quat(order), dtype=arr.dtype)
        default = xp.broadcast_to(default, arr.shape)
        return xp.where(n > eps, normalized, default)
    if zero_policy == "zero":
        return xp.where(n > eps, normalized, xp.zeros_like(arr))
    raise ValueError(f"Unsupported zero_policy '{zero_policy}'. Choose from: raise, identity, zero.")


def normalize_quat_xyzw(q, eps: float = 1e-8, zero_policy: str = "raise"):
    return _normalize_quat_impl(q, order="xyzw", eps=eps, zero_policy=zero_policy)


def normalize_quat_wxyz(q, eps: float = 1e-8, zero_policy: str = "raise"):
    return _normalize_quat_impl(q, order="wxyz", eps=eps, zero_policy=zero_policy)


def quat_xyzw_to_wxyz(quat):
    xp = _xp(quat)
    q = xp.asarray(quat, dtype=xp.float32 if xp is jnp else np.float32)
    return xp.concatenate([q[..., -1:], q[..., :3]], axis=-1)


def quat_wxyz_to_xyzw(quat):
    xp = _xp(quat)
    q = xp.asarray(quat, dtype=xp.float32 if xp is jnp else np.float32)
    return xp.concatenate([q[..., 1:], q[..., :1]], axis=-1)

def quw2wu(quw):
    return jnp.concatenate([quw[...,-1:], quw[...,:3]], axis=-1)

def qrand(outer_shape, jkey=None):
    if jkey is None:
        return qrand_np(outer_shape)
    else:
        return normalize(jax.random.normal(jkey, outer_shape + (4,)))

def qrand_np(outer_shape):
    q = np.random.normal(size=outer_shape+(4,))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q

def line2q(zaxis, yaxis=np.array([1,0,0])):
    Rm = line2Rm(zaxis, yaxis)
    return Rm2q(Rm)

def qmulti(q1, q2):
    b,c,d,a = jnp.split(q1, 4, axis=-1)
    f,g,h,e = jnp.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return jnp.concatenate([x,y,z,w], axis=-1)

def qmulti_np(q1, q2):
    b,c,d,a = np.split(q1, 4, axis=-1)
    f,g,h,e = np.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return np.concatenate([x,y,z,w], axis=-1)

def qinv(q):
    x,y,z,w = jnp.split(q, 4, axis=-1)
    return jnp.concatenate([-x,-y,-z,w], axis=-1)

def qinv_np(q):
    x,y,z,w = np.split(q, 4, axis=-1)
    return np.concatenate([-x,-y,-z,w], axis=-1)

def q2aa(q):
    return 2*qlog(q)[...,:3]

# def qlog(q):
#     # Clamp to avoid domain errors in arccos due to floating-point inaccuracies
#     q_w = jnp.clip(q[..., 3:], -1 + 1e-7, 1 - 1e-7)
    
#     # Compute alpha with clamped w-component
#     alpha = jnp.arccos(q_w)
#     sinalpha = jnp.sin(alpha)
    
#     # Ensure stable division by using a safe minimum threshold for sinalpha
#     safe_sinalpha = jnp.where(jnp.abs(sinalpha) < 1e-6, 1e-6, sinalpha)
#     n = q[..., :3] / (safe_sinalpha * jnp.sign(sinalpha))
    
#     # Use a threshold to check for small values of alpha
#     res = jnp.where(jnp.abs(q_w) < 1 - 1e-6, n * alpha, jnp.zeros_like(n))
    
#     # Concatenate result with an additional zero for the w-component
#     return jnp.concatenate([res, jnp.zeros_like(res[..., :1])], axis=-1)

def qlog(q, canonicalize_hemisphere=True, eps=1e-12):
    """
    Logarithm of a (unit) quaternion q = [x, y, z, w] -> [vx, vy, vz, 0].
    - Numerically stable for small angles (near identity) and avoids arccos clamping.
    - Uses atan2(norm(v), w) which behaves well near pi as well.
    - If canonicalize_hemisphere is True, flips q if w < 0 to avoid the antipodal
      discontinuity (better conditioning near 180°).
    """
    # Normalize (important if input is not exactly unit)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

    # Optional: flip to canonical hemisphere so w >= 0
    if canonicalize_hemisphere:
        sign = jnp.where(q[..., 3:4] < 0.0, -1.0, 1.0)
        q = q * sign

    v = q[..., :3]                # vector part
    w = q[..., 3]                 # scalar part
    s = jnp.linalg.norm(v, axis=-1, keepdims=True)  # ||v||

    # Stable angle: phi in [0, pi]
    # Works without any clamping and is well-conditioned near 1 and -1.
    phi = jnp.arctan2(s[..., 0], w)                  # shape (...,)

    # Compute v * (phi / s), but avoid 0/0 when s ~ 0.
    # For s->0: phi ~ s (if w~1), so phi/s -> 1 and result -> v (first-order accurate).
    # We use a safe factor = phi/s when s>eps, else 1.0.
    factor = jnp.where(s > eps, (phi[..., None] / s), jnp.ones_like(s))

    vec = v * factor  # (...,3)
    return jnp.concatenate([vec, jnp.zeros_like(w[..., None])], axis=-1)

def qLog(q):
    return qvee(qlog(q))

def qvee(phi):
    return 2*phi[...,:-1]

def qhat(w):
    return jnp.concatenate([w*0.5, jnp.zeros_like(w[...,0:1])], axis=-1)

def aa2q(aa):
    return qexp(aa*0.5)

def q2R(q):
    xp = _xp(q)
    arr = xp.asarray(q, dtype=xp.float32 if xp is jnp else np.float32)
    i, j, k, r = xp.split(arr, 4, axis=-1)
    R1 = xp.concatenate([1 - 2 * (j**2 + k**2), 2 * (i * j - k * r), 2 * (i * k + j * r)], axis=-1)
    R2 = xp.concatenate([2 * (i * j + k * r), 1 - 2 * (i**2 + k**2), 2 * (j * k - i * r)], axis=-1)
    R3 = xp.concatenate([2 * (i * k - j * r), 2 * (j * k + i * r), 1 - 2 * (i**2 + j**2)], axis=-1)
    return xp.stack([R1, R2, R3], axis=-2)

# def qexp(logq):
#     if isinstance(logq, np.ndarray):
#         alpha = np.linalg.norm(logq[...,:3], axis=-1, keepdims=True)
#         alpha = np.maximum(alpha, 1e-6)
#         return np.concatenate([logq[...,:3]/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)
#     else:
#         alpha = safe_norm(logq[...,:3], axis=-1, keepdims=True)
#         alpha = jnp.maximum(alpha, 1e-6)
#         return jnp.concatenate([logq[...,:3]/alpha*jnp.sin(alpha), jnp.cos(alpha)], axis=-1)


def _sinc(x, xp, thresh=1e-4):
    # sinc(x) = sin(x)/x with a stable polynomial near 0
    x2 = x * x
    # 1 - x^2/6 + x^4/120 is accurate to O(x^6)
    poly = 1.0 - x2 / 6.0 + (x2 * x2) / 120.0
    safe_denom = xp.where(xp.abs(x) > thresh, x, xp.ones_like(x))
    return xp.where(xp.abs(x) > thresh, xp.sin(x) / safe_denom, poly)

def qexp(logq, normalize_if_unit=True):
    """
    Quaternion exponential.

    Input:
      logq: (..., 3) or (..., 4)
            If (...,3): interpreted as [v] with scalar part s=0 (unit quaternion output).
            If (...,4): interpreted as [v, s] for general quaternion exp.

    Returns:
      q: (..., 4) quaternion in [x, y, z, w] layout.

    Numerics:
      - Uses sinc(x) small-angle series to avoid division by tiny norms.
      - Stable for angles near 0 and near pi.
      - If s=0 (pure rotation), output is unit; `normalize_if_unit=True` cleans tiny drift.
    """
    xp = _xp(logq)

    # split into vector and (optional) scalar part
    v = logq[..., :3]
    if logq.shape[-1] == 4:
        s = logq[..., 3:4]
    else:
        s = xp.zeros_like(logq[..., :1])  # s = 0 → unit quaternion output

    alpha = xp.linalg.norm(v, axis=-1, keepdims=True)        # ||v||
    sinc_alpha = _sinc(alpha, xp)                            # sin(alpha)/alpha with small-angle branch

    exp_s = xp.exp(s)                                        # handles general case (non-unit)
    vec = exp_s * v * sinc_alpha                             # e^s * v * (sin α / α)
    w   = exp_s * xp.cos(alpha)                              # e^s * cos α

    q = xp.concatenate([vec, w], axis=-1)

    # If we expect a unit quaternion (s ≈ 0), optional normalization removes round-off.
    if normalize_if_unit:
        # only normalize where |s| is tiny; avoid unnecessary work if you keep s nonzero
        mask = xp.abs(s) < 1e-12
        if mask.ndim > 0:  # broadcast-safe normalization
            nrm = xp.linalg.norm(q, axis=-1, keepdims=True)
            q = xp.where(mask, q / xp.maximum(nrm, 1e-30), q)
        else:
            if mask:
                nrm = xp.linalg.norm(q, axis=-1, keepdims=True)
                q = q / xp.maximum(nrm, 1e-30)

    return q

def pq_quatnormalize(pqc):
    return jnp.concatenate([pqc[...,:3], normalize(pqc[...,3:])], axis=-1)

def qExp(w):
    return qexp(qhat(w))

def qaction(quat, pos):
    return qmulti(qmulti(quat, jnp.concatenate([pos, jnp.zeros_like(pos[...,:1])], axis=-1)), qinv(quat))[...,:3]

def qaction_np(quat, pos):
    return qmulti_np(qmulti_np(quat, np.concatenate([pos, np.zeros_like(pos[...,:1])], axis=-1)), qinv_np(quat))[...,:3]

def qnoise(quat, scale=np.pi*10/180):
    lq = np.random.normal(scale=scale, size=quat[...,:3].shape)
    return qmulti(quat, qexp(lq))

def qzero(outer_shape):
    return jnp.concatenate([jnp.zeros(outer_shape + (3,)), jnp.ones(outer_shape + (1,))], axis=-1)

# posquat operations
def pq_inv(pos, quat=None):
    is_pqc = False
    if pos.shape[-1] == 7:
        is_pqc = True
        assert quat is None
        quat = pos[...,3:]
        pos = pos[...,:3]
    quat_inv = qinv(quat)
    if is_pqc:
        return jnp.concat([-qaction(quat_inv, pos), quat_inv], axis=-1)
    else:
        return -qaction(quat_inv, pos), quat_inv

def pq_action(translate, rotate, pnt=None):
    if translate.shape[-1] == 7:
        assert pnt is None
        assert rotate.shape[-1] == 3
        pnt = rotate
        pos = translate[...,:3]
        quat = translate[...,3:]
        return qaction(quat, pnt) + pos
    return qaction(rotate, pnt) + translate

def pq_multi(pos1, quat1, pos2=None, quat2=None):
    if pos1.shape[-1] == 7:
        assert quat1.shape[-1] == 7
        assert pos2 is None
        assert quat2 is None
        pos2 = quat1[...,:3]
        quat2 = quat1[...,3:]
        quat1 = pos1[...,3:]
        pos1 = pos1[...,:3]
        return jnp.concat([qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)], axis=-1)
    else:
        assert pos2 is not None
        assert quat2 is not None
        return qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)

def pqc_Exp(twist):
    return jnp.concat([twist[...,:3], qExp(twist[...,3:])], axis=-1)

def pqc_Log(pqc):
    return jnp.concat([pqc[...,:3], qLog(pqc[...,3:])], axis=-1)

def pqc_minus(pqc1, pqc2, output_as_global_orientation=False):
    '''
    pqc1 - pqc2
    '''
    if pqc1.shape[-1] != 7:
        # only position
        return pqc1 - pqc2
    quat1 = pqc1[...,3:]
    quat2 = pqc2[...,3:]
    flip_mask = jnp.linalg.norm(quat1 - quat2, axis=-1, keepdims=True) > jnp.linalg.norm(-quat1 - quat2, axis=-1, keepdims=True)
    pqc1 = jnp.where(flip_mask, jnp.concat([pqc1[...,:3], -pqc1[...,3:]], axis=-1), pqc1)
    pqc_exp = pq_multi(pq_inv(pqc2), pqc1)
    pqc_exp = pq_quatnormalize(pqc_exp)
    if output_as_global_orientation:
        return se3_rot(pqc_Log(pqc_exp), pqc2[...,3:])
    else:
        return pqc_Log(pqc_exp)


def pq2H(pos, quat=None):
    if pos.shape[-1] == 7:
        assert quat is None
        quat = pos[...,-4:]
        pos = pos[...,:3]
    else:
        assert quat is not None

    R = q2R(quat)
    return H_from_Rpos(R, pos)

def pq_wfirst_to_wlast(pqc):
    """
    Convert pqc from [pos, quat_wfirst] to [pos, quat_wlast].
    """
    pos = pqc[..., :3]
    quat_wfirst = pqc[..., 3:7]
    remaining = pqc[..., 7:]
    quat_wlast = jnp.concatenate([quat_wfirst[..., 1:], quat_wfirst[..., :1]], axis=-1)
    return jnp.concatenate([pos, quat_wlast, remaining], axis=-1)

def pq_wlast_to_wfirst(pqc):
    """
    Convert pqc from [pos, quat_wlast] to [pos, quat_wfirst].
    """
    pos = pqc[..., :3]
    quat_wlast = pqc[..., 3:7]
    remaining = pqc[..., 7:]
    quat_wfirst = jnp.concatenate([quat_wlast[..., -1:], quat_wlast[..., :-1]], axis=-1)
    return jnp.concatenate([pos, quat_wfirst, remaining], axis=-1)

def pq_quat_equiv_flip(pqc):
    return jnp.where(pqc[..., 6:7] > 0, pqc, jnp.concatenate([pqc[...,:3], -pqc[...,3:]], axis=-1))

def quat_dot_to_omega(quat, quat_dot):
    w_body = qvee(qmulti(qinv(quat), quat_dot))
    w_global = qaction(quat, w_body)
    # w_global = qvee(qmulti(quat_dot, qinv(quat)))
    return w_global

def pq_dot_to_se3(pqc, pqc_dot):
    pos_dot = pqc_dot[...,:3]
    quat_dot = pqc_dot[...,3:]
    quat = pqc[...,3:]
    w_global = quat_dot_to_omega(quat, quat_dot)
    return jnp.concatenate([pos_dot, w_global], axis=-1)


def apply_pq_to_se3(pqc_AB, se3_B, convention='vw'):
    """
    pqc_AB: transform from frame A to frame B, H^A_B
    assume that se3 is expressed in the local frame
    """
    if convention == 'vw':
        wA = qaction(pqc_AB[...,3:], se3_B[...,3:])
        vA = qaction(pqc_AB[...,3:], se3_B[...,:3]) + jnp.cross(pqc_AB[...,:3], qaction(pqc_AB[...,3:], se3_B[...,3:]))
        return jnp.concatenate([vA, wA], axis=-1)
    elif convention == 'wv':
        wA = qaction(pqc_AB[...,3:], se3_B[...,:3])
        vA = qaction(pqc_AB[...,3:], se3_B[...,3:]) + jnp.cross(pqc_AB[...,:3], qaction(pqc_AB[...,3:], se3_B[...,:3]))
        return jnp.concatenate([wA, vA], axis=-1)

def transform_spatial_vel(pqc_A, pqc_B, spatial_vel_B, vel_as_global_orientation=False, output_as_global_orientation=False, convention='wv'):
    if vel_as_global_orientation:
        spatial_vel_B = se3_rot(spatial_vel_B, qinv(pqc_B[...,3:]))
    transform_AB = pq_multi(pq_inv(pqc_A), pqc_B)
    if output_as_global_orientation:
        return se3_rot(apply_pq_to_se3(transform_AB, spatial_vel_B, convention=convention), pqc_A[...,3:])
    return apply_pq_to_se3(transform_AB, spatial_vel_B, convention=convention)


# homogineous transforms
def H_from_Rpos(R, pos):
    H = jnp.zeros(pos.shape[:-1] + (4,4))
    H = H.at[...,-1,-1].set(1)
    H = H.at[...,:3,:3].set(R)
    H = H.at[...,:3,3].set(pos)
    return H

def H_inv(H):
    R = H[...,:3,:3]
    p = H[...,:3, 3:]
    return H_from_Rpos(T(R), (-T(R)@p)[...,0])

def H2pq(H, concat=False):
    Rm = H[...,:3,:3]
    pos = H[...,:3, 3]
    if concat:
        return jnp.concatenate([pos, Rm2q(Rm)], axis=-1)
    else:
        return pos, Rm2q(Rm)

# Rm util
def Rm_inv(Rm):
    return T(Rm)

# def line2Rm(zaxis, yaxis=np.array([1,0,0])):
#     valid_mask = jnp.linalg.norm(zaxis, axis=-1, keepdims=True) > 1e-8
#     zaxis = normalize(zaxis)
#     xaxis = jnp.cross(yaxis, zaxis)
#     xaxis = normalize(xaxis)
#     yaxis = jnp.cross(zaxis, xaxis)
#     Rm = jnp.stack([xaxis, yaxis, zaxis], axis=-1)
#     Rm = jnp.where(valid_mask[..., None], Rm, jnp.eye(3))
#     return Rm


def line2Rm(zaxis: jnp.ndarray, y_hint: jnp.ndarray | None = None, eps: float = 1e-12) -> jnp.ndarray:
    """
    Build a right-handed rotation matrix R from a given z-axis direction.
    Optionally uses y_hint as a reference; otherwise picks a safe auxiliary axis.
    Shapes:
      zaxis: (..., 3)
      y_hint: (..., 3) or None
    Returns:
      R: (..., 3, 3) whose columns are [x, y, z].
    """
    xp = _xp(zaxis)
    # Valid z?
    z_valid = safe_norm(zaxis, axis=-1, keepdims=True, eps=eps) > 1e-8
    z = normalize(zaxis, axis=-1, eps=eps)

    # If no y_hint given, choose an auxiliary axis that is least aligned with z
    # Heuristic: pick the standard basis vector corresponding to the smallest |z| component
    if y_hint is None:
        # basis vectors broadcastable to z
        e1 = xp.array([1.0, 0.0, 0.0], dtype=z.dtype)
        e2 = xp.array([0.0, 1.0, 0.0], dtype=z.dtype)
        e3 = xp.array([0.0, 0.0, 1.0], dtype=z.dtype)
        absz = xp.abs(z)
        # index of smallest component per batch
        idx = xp.argmin(absz, axis=-1)
        # select e1/e2/e3 per idx
        a = xp.where(idx[..., None] == 0, e1,
            xp.where(idx[..., None] == 1, e2, e3))
    else:
        # Use provided hint
        a = y_hint

    # Project a to be orthogonal to z; if it nearly vanishes, switch to automatic auxiliary
    a_proj = a - xp.sum(a * z, axis=-1, keepdims=True) * z
    a_proj_norm = safe_norm(a_proj, axis=-1, keepdims=True, eps=eps)
    need_fallback = a_proj_norm[..., 0] < 1e-6

    # Fallback auxiliary if hint is collinear
    e1 = xp.array([1.0, 0.0, 0.0], dtype=z.dtype)
    e2 = xp.array([0.0, 1.0, 0.0], dtype=z.dtype)
    e3 = xp.array([0.0, 0.0, 1.0], dtype=z.dtype)
    absz = xp.abs(z)
    idx_fb = xp.argmin(absz, axis=-1)
    a_fb = xp.where(idx_fb[..., None] == 0, e1,
             xp.where(idx_fb[..., None] == 1, e2, e3))
    a_use = xp.where(need_fallback[..., None], a_fb, a)
    a_use = a_use - xp.sum(a_use * z, axis=-1, keepdims=True) * z
    a_use = normalize(a_use, axis=-1, eps=eps)

    # Now build an orthonormal, right-handed frame:
    # First x' = normalized (a_use × z)
    x_prime = normalize(xp.cross(a_use, z), axis=-1, eps=eps)
    # Then y = normalized (z × x')
    y = normalize(xp.cross(z, x_prime), axis=-1, eps=eps)
    # Recompute x = y × z to kill any residual non-orthogonality
    x = xp.cross(y, z)

    # Stack columns [x y z]
    R = xp.stack([x, y, z], axis=-1)

    # If z was invalid, return identity
    I = xp.eye(3, dtype=R.dtype)
    R = xp.where(z_valid[..., None], R, I)
    return R

# def line2Rm_np(zaxis, yaxis=np.array([1,0,0])):
#     zaxis = (zaxis + np.array([0,1e-6,0]))
#     zaxis = zaxis/np.linalg.norm(zaxis, axis=-1, keepdims=True)
#     xaxis = np.cross(yaxis, zaxis)
#     xaxis = xaxis/np.linalg.norm(xaxis, axis=-1, keepdims=True)
#     yaxis = np.cross(zaxis, xaxis)
#     Rm = np.stack([xaxis, yaxis, zaxis], axis=-1)
#     return Rm

def Rm2q(Rm):
    xp = _xp(Rm)
    arr = xp.asarray(Rm, dtype=xp.float32 if xp is jnp else np.float32)
    Rm = xp.swapaxes(arr, -1, -2)
    con1 = (Rm[...,2,2] < 0) & (Rm[...,0,0] > Rm[...,1,1])
    con2 = (Rm[...,2,2] < 0) & (Rm[...,0,0] <= Rm[...,1,1])
    con3 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] < -Rm[...,1,1])
    con4 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] >= -Rm[...,1,1]) 

    t1 = 1 + Rm[...,0,0] - Rm[...,1,1] - Rm[...,2,2]
    t2 = 1 - Rm[...,0,0] + Rm[...,1,1] - Rm[...,2,2]
    t3 = 1 - Rm[...,0,0] - Rm[...,1,1] + Rm[...,2,2]
    t4 = 1 + Rm[...,0,0] + Rm[...,1,1] + Rm[...,2,2]

    q1 = xp.stack([t1, Rm[...,0,1]+Rm[...,1,0], Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]-Rm[...,2,1]], axis=-1) / xp.sqrt(xp.maximum(t1, 1e-7))[...,None]
    q2 = xp.stack([Rm[...,0,1]+Rm[...,1,0], t2, Rm[...,1,2]+Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2]], axis=-1) / xp.sqrt(xp.maximum(t2, 1e-7))[...,None]
    q3 = xp.stack([Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]+Rm[...,2,1], t3, Rm[...,0,1]-Rm[...,1,0]], axis=-1) / xp.sqrt(xp.maximum(t3, 1e-7))[...,None]
    q4 = xp.stack([Rm[...,1,2]-Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2], Rm[...,0,1]-Rm[...,1,0], t4], axis=-1) / xp.sqrt(xp.maximum(t4, 1e-7))[...,None]
 
    q = xp.zeros(Rm.shape[:-2]+(4,), dtype=Rm.dtype)
    q = xp.where(con1[...,None], q1, q)
    q = xp.where(con2[...,None], q2, q)
    q = xp.where(con3[...,None], q3, q)
    q = xp.where(con4[...,None], q4, q)
    q *= 0.5

    return q


def quat_xyzw_to_matrix(quat):
    q = normalize_quat_xyzw(quat, zero_policy="identity")
    R = q2R(q)
    if _is_jax_array(q):
        return jnp.asarray(R, dtype=jnp.float32)
    return np.asarray(R, dtype=np.float32)


def quat_wxyz_to_matrix(quat):
    return quat_xyzw_to_matrix(quat_wxyz_to_xyzw(quat))


def matrix_to_quat_xyzw(rot):
    xp = _xp(rot)
    rot_arr = xp.asarray(rot, dtype=xp.float32 if xp is jnp else np.float32)
    quat = normalize_quat_xyzw(Rm2q(rot_arr), zero_policy="identity")
    if _is_jax_array(rot):
        return quat
    return np.asarray(quat, dtype=np.float32)


def matrix_to_quat_wxyz(rot):
    quat = quat_xyzw_to_wxyz(matrix_to_quat_xyzw(rot))
    if _is_jax_array(rot):
        return quat
    return np.asarray(quat, dtype=np.float32)


def quat_mul_xyzw(q1, q2):
    if _is_jax_array(q1) or _is_jax_array(q2):
        return normalize_quat_xyzw(qmulti(jnp.asarray(q1, dtype=jnp.float32), jnp.asarray(q2, dtype=jnp.float32)), zero_policy="identity")
    return np.asarray(
        normalize_quat_xyzw(qmulti_np(np.asarray(q1, dtype=np.float32), np.asarray(q2, dtype=np.float32)), zero_policy="identity"),
        dtype=np.float32,
    )


def quat_mul_wxyz(q1, q2):
    q = quat_xyzw_to_wxyz(quat_mul_xyzw(quat_wxyz_to_xyzw(q1), quat_wxyz_to_xyzw(q2)))
    if _is_jax_array(q1) or _is_jax_array(q2):
        return q
    return np.asarray(q, dtype=np.float32)


def quat_inv_xyzw(q):
    q_norm = normalize_quat_xyzw(q, zero_policy="identity")
    if _is_jax_array(q_norm):
        return qinv(jnp.asarray(q_norm, dtype=jnp.float32))
    return np.asarray(qinv_np(np.asarray(q_norm, dtype=np.float32)), dtype=np.float32)


def quat_inv_wxyz(q):
    quat = quat_xyzw_to_wxyz(quat_inv_xyzw(quat_wxyz_to_xyzw(q)))
    if _is_jax_array(q):
        return quat
    return np.asarray(quat, dtype=np.float32)


def quat_apply_xyzw(quat, points):
    if _is_jax_array(quat) or _is_jax_array(points):
        return qaction(jnp.asarray(quat, dtype=jnp.float32), jnp.asarray(points, dtype=jnp.float32))
    return np.asarray(qaction_np(np.asarray(quat, dtype=np.float32), np.asarray(points, dtype=np.float32)), dtype=np.float32)


def quat_apply_wxyz(quat, points):
    return quat_apply_xyzw(quat_wxyz_to_xyzw(quat), points)


def rotvec_to_matrix(axis_angle):
    axis_angle_arr = _xp(axis_angle).asarray(axis_angle, dtype=jnp.float32 if _is_jax_array(axis_angle) else np.float32)
    return quat_xyzw_to_matrix(aa2q(axis_angle_arr))


def axis_angle_to_matrix(axis, angle):
    xp = _xp(axis)
    axis_arr = xp.asarray(axis, dtype=xp.float32 if xp is jnp else np.float32)
    angle_arr = xp.asarray(angle, dtype=axis_arr.dtype)
    norm = xp.linalg.norm(axis_arr, axis=-1, keepdims=True)
    safe_axis = xp.where(norm > 1e-8, axis_arr / xp.maximum(norm, 1e-8), axis_arr)
    return rotvec_to_matrix(safe_axis * angle_arr[..., None])


def _homogeneous_from_rotation_translation(rot, pos):
    xp = _xp(rot)
    rot_arr = xp.asarray(rot, dtype=xp.float32 if xp is jnp else np.float32)
    pos_arr = xp.asarray(pos, dtype=rot_arr.dtype)
    out_shape = rot_arr.shape[:-2] + (4, 4)
    if xp is jnp:
        H = xp.broadcast_to(xp.eye(4, dtype=rot_arr.dtype), out_shape)
        H = H.at[..., :3, :3].set(rot_arr)
        H = H.at[..., :3, 3].set(pos_arr)
        return H
    H = np.broadcast_to(np.eye(4, dtype=np.float32), out_shape).copy()
    H[..., :3, :3] = np.asarray(rot_arr, dtype=np.float32)
    H[..., :3, 3] = np.asarray(pos_arr, dtype=np.float32)
    return H.astype(np.float32)


def pose_xyzw_to_matrix(pos, quat):
    return _homogeneous_from_rotation_translation(quat_xyzw_to_matrix(quat), pos)


def pose_wxyz_to_matrix(pos, quat):
    return _homogeneous_from_rotation_translation(quat_wxyz_to_matrix(quat), pos)


def matrix_to_pose_xyzw(H):
    H_arr = jnp.asarray(H, dtype=jnp.float32) if _is_jax_array(H) else np.asarray(H, dtype=np.float32)
    pos = H_arr[..., :3, 3]
    quat = matrix_to_quat_xyzw(H_arr[..., :3, :3])
    if _is_jax_array(H):
        return pos, quat
    return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)


def matrix_to_pose_wxyz(H):
    pos, quat = matrix_to_pose_xyzw(H)
    quat_wxyz = quat_xyzw_to_wxyz(quat)
    if _is_jax_array(H):
        return pos, quat_wxyz
    return np.asarray(pos, dtype=np.float32), np.asarray(quat_wxyz, dtype=np.float32)


def _align_frame_values_to_points(points, frame_values, *, value_axes: int):
    xp = _xp(frame_values)
    points_leading_ndim = points.ndim - 1
    frame_leading_ndim = frame_values.ndim - value_axes
    extra_dims = max(points_leading_ndim - frame_leading_ndim, 0)
    axis = -(value_axes + 1)
    for _ in range(extra_dims):
        frame_values = xp.expand_dims(frame_values, axis=axis)
    return frame_values


def points_local_to_world(points_local, frame_pos_world, frame_rot_world):
    use_jax = _is_jax_array(points_local) or _is_jax_array(frame_pos_world) or _is_jax_array(frame_rot_world)
    pts = jnp.asarray(points_local, dtype=jnp.float32) if use_jax else np.asarray(points_local, dtype=np.float32)
    pos = jnp.asarray(frame_pos_world, dtype=jnp.float32) if use_jax else np.asarray(frame_pos_world, dtype=np.float32)
    rot = jnp.asarray(frame_rot_world, dtype=jnp.float32) if use_jax else np.asarray(frame_rot_world, dtype=np.float32)
    pos = _align_frame_values_to_points(pts, pos, value_axes=1)
    rot = _align_frame_values_to_points(pts, rot, value_axes=2)
    if use_jax:
        out = jnp.einsum("...i,...ji->...j", pts, rot) + pos
    else:
        out = np.einsum("...i,...ji->...j", pts, rot, optimize=True) + pos
    if _is_jax_array(out):
        return out
    return np.asarray(out, dtype=np.float32)


def points_world_to_local(points_world, frame_pos_world, frame_rot_world):
    use_jax = _is_jax_array(points_world) or _is_jax_array(frame_pos_world) or _is_jax_array(frame_rot_world)
    pts = jnp.asarray(points_world, dtype=jnp.float32) if use_jax else np.asarray(points_world, dtype=np.float32)
    pos = jnp.asarray(frame_pos_world, dtype=jnp.float32) if use_jax else np.asarray(frame_pos_world, dtype=np.float32)
    rot = jnp.asarray(frame_rot_world, dtype=jnp.float32) if use_jax else np.asarray(frame_rot_world, dtype=np.float32)
    pos = _align_frame_values_to_points(pts, pos, value_axes=1)
    rot = _align_frame_values_to_points(pts, rot, value_axes=2)
    if use_jax:
        out = jnp.einsum("...i,...ij->...j", pts - pos, rot)
    else:
        out = np.einsum("...i,...ij->...j", pts - pos, rot, optimize=True)
    if _is_jax_array(out):
        return out
    return np.asarray(out, dtype=np.float32)


def compose_pose(parent_pos_world, parent_rot_world, child_pos_local, child_rot_local):
    pos = points_local_to_world(child_pos_local, parent_pos_world, parent_rot_world)
    use_jax = _is_jax_array(parent_pos_world) or _is_jax_array(parent_rot_world) or _is_jax_array(child_pos_local) or _is_jax_array(child_rot_local)
    rot_a = jnp.asarray(parent_rot_world, dtype=jnp.float32) if use_jax else np.asarray(parent_rot_world, dtype=np.float32)
    rot_b = jnp.asarray(child_rot_local, dtype=jnp.float32) if use_jax else np.asarray(child_rot_local, dtype=np.float32)
    rot = rot_a @ rot_b
    if _is_jax_array(rot):
        return pos, rot
    return np.asarray(pos, dtype=np.float32), np.asarray(rot, dtype=np.float32)


def invert_pose(pos_world, rot_world):
    use_jax = _is_jax_array(pos_world) or _is_jax_array(rot_world)
    rot = jnp.asarray(rot_world, dtype=jnp.float32) if use_jax else np.asarray(rot_world, dtype=np.float32)
    pos = jnp.asarray(pos_world, dtype=jnp.float32) if use_jax else np.asarray(pos_world, dtype=np.float32)
    rot_inv = jnp.swapaxes(rot, -1, -2) if _is_jax_array(rot) else np.swapaxes(rot, -1, -2)
    pos_inv = -(pos @ rot)
    if _is_jax_array(rot_inv):
        return pos_inv, rot_inv
    return np.asarray(pos_inv, dtype=np.float32), np.asarray(rot_inv, dtype=np.float32)


def pose_local_to_world(local_pos, local_rot, parent_pos_world, parent_rot_world):
    return compose_pose(parent_pos_world, parent_rot_world, local_pos, local_rot)


def pose_world_to_local(world_pos, world_rot, frame_pos_world, frame_rot_world):
    pos = points_world_to_local(world_pos, frame_pos_world, frame_rot_world)
    use_jax = _is_jax_array(world_pos) or _is_jax_array(world_rot) or _is_jax_array(frame_pos_world) or _is_jax_array(frame_rot_world)
    rot_a = jnp.asarray(frame_rot_world, dtype=jnp.float32) if use_jax else np.asarray(frame_rot_world, dtype=np.float32)
    rot_b = jnp.asarray(world_rot, dtype=jnp.float32) if use_jax else np.asarray(world_rot, dtype=np.float32)
    rot_local = (jnp.swapaxes(rot_a, -1, -2) if _is_jax_array(rot_a) else np.swapaxes(rot_a, -1, -2)) @ rot_b
    if _is_jax_array(rot_local):
        return pos, rot_local
    return np.asarray(pos, dtype=np.float32), np.asarray(rot_local, dtype=np.float32)

def pRm_inv(pos, Rm):
    # return (-T(Rm)@pos[...,None,:])[...,0], T(Rm)
    return jnp.einsum('...ij,...j->...i', T(Rm), -pos), T(Rm)

def pRm_action(pos, Rm, x):
    # return (Rm @ x[...,None,:])[...,0] + pos
    return jnp.einsum('...ij,...j->...i', Rm, x) + pos

def pRm2pq(pos, Rm):
    return jnp.concatenate([pos, Rm2q(Rm)], axis=-1)

def se3_rot(se3, quat):
    if se3.shape[-1] == 3:
        # only position
        return se3
    return jnp.concat([qaction(quat, se3[...,:3]), qaction(quat, se3[...,3:])], axis=-1)


# 6d utils
def R6d2Rm(x, gram_schmidt=False):
    xv, yv = x[...,:3], x[...,3:]
    xv = normalize(xv)
    if gram_schmidt:
        yv = normalize(yv - jnp.einsum('...i,...i',yv,xv)[...,None]*xv)
        zv = jnp.cross(xv, yv)
    else:
        zv = jnp.cross(xv, yv)
        zv = normalize(zv)
        yv = jnp.cross(zv, xv)
    return jnp.stack([xv,yv,zv], -1)

# 9d utils
def R9d2Rm(x):
    xm = einops.rearrange(x, '... (t i) -> ... t i', t=3)
    u, s, vt = jnp.linalg.svd(xm)
    # vt = einops.rearrange(v, '... i j -> ... j i')
    det = jnp.linalg.det(jnp.matmul(u,vt))
    vtn = jnp.concatenate([vt[...,:2,:], vt[...,2:,:]*det[...,None,None]], axis=-2)
    return jnp.matmul(u,vtn)


# general
def T(mat):
    return einops.rearrange(mat, '... i j -> ... j i')

def pq2SE2h(pos, quat=None):
    if pos.shape[-1] == 7:
        assert quat is None
        quat = pos[...,-4:]
        pos = pos[...,:3]
    z_angle = q2aa(quat)[...,2]
    SE2 = jnp.concat([pos[...,:2], z_angle[...,None]], axis=-1)
    height = pos[...,2]
    return SE2, height

def SE2h2pq(SE2, height, concat=False):
    height = jnp.array(height)
    pos = jnp.concatenate([SE2[...,:2], height[...,None]], axis=-1)
    quat = aa2q(jnp.concatenate([jnp.zeros_like(SE2[...,:2]), SE2[...,2:]], axis=-1))
    if concat:
        return jnp.concat([pos, quat], axis=-1)
    else:
        return pos, quat

# euler angle
def Rm2ZYZeuler(Rm):
    sy = jnp.sqrt(Rm[...,0,2]**2+Rm[...,1,2]**2)
    v1 = jnp.arctan2(Rm[...,1,2], Rm[...,0,2])
    v2 = jnp.arctan2(sy, Rm[...,2,2])
    v3 = jnp.arctan2(Rm[...,2,1], -Rm[...,2,0])

    v1n = jnp.arctan2(-Rm[...,0,1], Rm[...,1,1])
    v1 = jnp.where(sy < 1e-6, v1n, v1)
    v3 = jnp.where(sy < 1e-6, jnp.zeros_like(v1), v3)

    return jnp.stack([v1,v2,v3],-1)

def Rm2YXYeuler(Rm):
    sy = jnp.sqrt(jnp.sqrt(Rm[...,0,1]**2+Rm[...,2,1]**2))
    v1 = jnp.arctan2(Rm[...,0,1], Rm[...,2,1])
    v2 = jnp.arctan2(sy, Rm[...,1,1])
    v3 = jnp.arctan2(Rm[...,1,0], -Rm[...,1,2])

    v1n = jnp.arctan2(-Rm[...,2,0], Rm[...,0,0])
    v1 = jnp.where(sy < 1e-6, v1n, v1)
    v3 = jnp.where(sy < 1e-6, jnp.zeros_like(v1), v3)

    return jnp.stack([v1,v2,v3],-1)

def YXYeuler2Rm(YXYeuler):
    c1,c2,c3 = jnp.split(jnp.cos(YXYeuler), 3, -1)
    s1,s2,s3 = jnp.split(jnp.sin(YXYeuler), 3, -1)
    return jnp.stack([jnp.concatenate([c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],-1),
            jnp.concatenate([s2*s3, c2, -c3*s2],-1),
            jnp.concatenate([-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3],-1)], -2)

def wigner_D_order1_from_Rm(Rm):
    r1,r2,r3 = jnp.split(Rm,3,-2)
    r11,r12,r13 = jnp.split(r1,3,-1)
    r21,r22,r23 = jnp.split(r2,3,-1)
    r31,r32,r33 = jnp.split(r3,3,-1)

    return jnp.concatenate([jnp.c_[r22, r23, r21],
                jnp.c_[r32, r33, r31],
                jnp.c_[r12, r13, r11]], axis=-2)

def q2ZYZeuler(q):
    return Rm2ZYZeuler(q2R(q))

def q2XYZeuler(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = jnp.split(q, 4, -1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = jnp.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = jnp.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = jnp.arctan2(t3, t4)
    
    return jnp.concatenate([roll_x, pitch_y, yaw_z], -1) # in radians

def XYZeuler2q(euler):
    """
    Convert euler angles (roll, pitch, yaw) to quaternion
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    roll_x, pitch_y, yaw_z = jnp.split(euler, 3, -1)
    cy = jnp.cos(yaw_z * 0.5)
    sy = jnp.sin(yaw_z * 0.5)
    cp = jnp.cos(pitch_y * 0.5)
    sp = jnp.sin(pitch_y * 0.5)
    cr = jnp.cos(roll_x * 0.5)
    sr = jnp.sin(roll_x * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return jnp.concatenate([x, y, z, w], -1)

def skew(v):
    """
    Create a skew-symmetric matrix from a vector.
    Args:
        v: A vector of shape (..., 3).
    Returns:
        A skew-symmetric matrix of shape (..., 3, 3).
    """
    assert v.shape[-1] == 3
    vx, vy, vz = jnp.split(v, 3, axis=-1)
    zero = jnp.zeros_like(vx)

    row0 = jnp.concatenate([ zero, -vz,  vy], axis=-1)
    row1 = jnp.concatenate([  vz,  zero, -vx], axis=-1)
    row2 = jnp.concatenate([-vy,   vx,  zero], axis=-1)

    return jnp.stack([row0, row1, row2], axis=-2)

def unskew(skew_matrix):
    """
    Extract a vector from a skew-symmetric matrix.
    Args:
        skew_matrix: A skew-symmetric matrix of shape (..., 3, 3).
    Returns:
        A vector of shape (..., 3).
    """
    assert skew_matrix.shape[-2:] == (3, 3)
    return jnp.stack([skew_matrix[..., 2, 1], skew_matrix[..., 0, 2], skew_matrix[..., 1, 0]], axis=-1)

def pq2ST(pos, quat=None):
    """
    Convert position and quaternion to a spatial transform vector.
    Args:
        pos: Position vector (..., 3).
        quat: Quaternion vector (..., 4).
    Returns:
        A 6D vector (..., 6, 6) representing the spatial transform.
    """
    if pos.shape[-1] == 7:
        assert quat is None
        quat = pos[...,-4:]
        pos = pos[...,:3]
    else:
        assert quat is not None

    assert pos.shape[-1] == 3
    assert quat.shape[-1] == 4

    Rm    = q2R(quat)                     # (...,3,3)
    p_sk  = skew(pos)              # (...,3,3)

    return jnp.block([
        [Rm, jnp.zeros_like(Rm)],
        [jnp.einsum('...ij,...jk->...ik', p_sk, Rm), Rm]
    ])

def ST2pq(ST):
    Rm = ST[...,:3,:3]
    Rmp = ST[...,3:6,:3]
    psk = jnp.einsum('...ij,...jk', Rmp, T(Rm))
    pos, quat = unskew(psk), Rm2q(Rm)
    posquat = jnp.concatenate([pos, quat], axis=-1)
    return posquat

def ST2forcedual(ST):
    Rm = ST[...,:3,:3]
    Rmp = ST[...,3:6,:3]
    zero33 = jnp.zeros_like(Rm)

    return jnp.block([[Rm, Rmp],[zero33, Rm]])


def spatial_inertia_matrix(
    mass,
    com,
    inertia_diag,
    inertial_quat,
):
    """
        Build a 6x6 spatial inertia matrix, which can be used by the Articulated Body Algorithm.
        Args:
            mass: mass of the body (...,)
            com: center of mass of the body (..., 3)
            inertia_diag: diagonal elements of the inertia tensor (..., 3)
            inertial_quat: quaternion representing the orientation of the body (..., 4)
        Returns
        -------
        I_sp : (..., 6,6) JAX array
            Spatial inertia matrix.
    """

    # outer_shape = mass.shape[:-1]
    outer_shape = mass.shape
    assert com.shape == outer_shape + (3,)
    assert inertia_diag.shape == outer_shape + (3,) or inertia_diag.shape == outer_shape + (3,3)
    assert inertial_quat.shape == outer_shape + (4,) or inertial_quat.shape == outer_shape + (3,3)

    # Convert quaternion to rotation matrix
    if inertial_quat.shape == outer_shape + (4,):
        R_I = q2R(inertial_quat)
    else:
        R_I = inertial_quat  # Assume it's already a rotation matrix
    if inertia_diag.shape == outer_shape + (3,3):
        inertia_matrix = inertia_diag
    elif inertia_diag.shape == outer_shape + (3,):
        inertia_matrix = jnp.diag(inertia_diag)
    # Ic = R_I @ inertia_matrix @ T(R_I)
    Ic = jnp.einsum('...ij,...jk,...lk->...il', R_I, inertia_matrix, R_I)
    c_sk = skew(com)

    mass_ext = mass[...,None,None]
    mI = mass_ext * jnp.eye(3, dtype=mass.dtype)
    I_sp = jnp.block([
        [Ic + mass_ext * c_sk @ T(c_sk), mass_ext * c_sk],
        [mass_ext * T(c_sk), mI]
    ])

    return I_sp

def crm(v):
    w, vlin = v[..., :3], v[..., 3:]
    wx = skew(w)
    vx = skew(vlin)
    upper = jnp.concatenate([wx, jnp.zeros_like(wx)], axis=-1)
    lower = jnp.concatenate([vx, wx], axis=-1)
    return jnp.concatenate([upper, lower], axis=-2)   # (6,6)

def crf(v):
    return T(-crm(v))  # (6,6)

def xjvj_joint_type(is_revolute, axis, qi, qdi):
    """
    Compute the joint type cross product operator.
    Args:
        is_revolute: Boolean indicating if the joint is revolute. True for revolute joints, False for prismatic joints.
        axis: Joint axis (3D vector). (..., 3).
        qi: Joint position (scalar). (...,)
        qdi: Joint velocity (scalar). (...,)
    Returns:
        A 6x6 matrix representing the joint type cross product operator.
        A 6D vector representing the joint velocity.
    """
    qi = qi[..., None]  # (..., 1)
    qdi = qdi[..., None]  # (..., 1)
    quat = aa2q(axis * qi) # (..., 4)
    XJ = jnp.where(
        is_revolute,
        pq2ST(jnp.zeros_like(axis), quat),
        pq2ST(axis * qi, jnp.zeros_like(quat)),
    )
    vJ = jnp.where(
        is_revolute,
        jnp.concatenate([axis * qdi, jnp.zeros_like(axis)], axis=-1),
        jnp.concatenate([jnp.zeros_like(axis), axis * qdi], axis=-1),
    )
    return XJ, vJ

def inverse_spation_transform(X):
    R = X[..., :3, :3]
    pR = X[..., 3:, :3]
    X_inv = X.at[...,3:,:3].set(T(pR))
    X_inv = X_inv.at[...,:3, 3:].set(0.0)
    X_inv = X_inv.at[..., :3, :3].set(T(R))
    X_inv = X_inv.at[..., 3:, 3:].set(T(R))
    return X_inv

if __name__ == "__main__":
    import pinocchio as pin

    def pin_pq2ST(pqc):
        H1 = pq2H(pqc)
        H1 = np.array(H1)
        M_AB = pin.SE3(H1[:3,:3], H1[:3,3])
        X_AB = M_AB.toDualActionMatrix()
        X_AB_star = M_AB.toActionMatrix()
        x_AB_star_2 = T(inverse_spation_transform(jnp.array(X_AB)))
        dual_mat = ST2forcedual(pq2ST(pqc))
        return X_AB

    jkey = jax.random.PRNGKey(0)
    jkey, sk1, sk2 = jax.random.split(jkey, 3)
    pqc1 = qrand((), sk1)
    pqc1 = jnp.concat([jax.random.uniform(sk2, shape=(3,)), pqc1], axis=-1)
    jkey, sk1, sk2 = jax.random.split(jkey, 3)
    pqc2 = qrand((), sk1)
    pqc2 = jnp.concat([jax.random.uniform(sk2, shape=(3,)), pqc2], axis=-1)

    X1 = pq2ST(pqc1)
    X2 = pq2ST(pqc2)

    X1_pin = pin_pq2ST(pqc1)
    X2_pin = pin_pq2ST(pqc2)

    X12 = X1@X2 # X3-1
    pqc12_rec = ST2pq(X12)
    pqc12 = pq_multi(pqc1, pqc2)
    X12_rec = pq2ST(pqc12)

    test_v = jnp.array([0,0,1,0,0,0.])

    # pqc1 = jnp.array([1, 0, 0, 0, 0, 0, 1.])
    pqc1 = jnp.array([1, 0, 0, np.sqrt(1/2), 0, 0, np.sqrt(1/2)])
    # pqc2 = jnp.array([4, 5, 6, 0, 0, 0, 1.])

    X1 = pq2ST(pqc1)

    pqc1_rec = ST2pq(X1)

    res = jnp.einsum('...ij,...j', inverse_spation_transform(X1), test_v) #answer should be [0, 1, 0, 0, 0, -1]

    print(1)
