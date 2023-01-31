import numpy as np
from utils.quater import *
import jax
import jax.numpy as jnp
import sys
from jax import jit
sys.path.append("..")


def motion(qt: np.array, omega: np.array, interval: np.float32) -> jnp.array:
    """
    Calculate the quaternion kinematics motion model

    Args:
        qt: quaternion, represented by np array
        omega: the anguler velocity detected by IMU
        interval: the interval between this measurement and the last measurement

    Returns:
        f: calculated motion model, which is q_(t+1)
    """
    expo = jnp.concatenate([jnp.array([0]), interval*omega/2])
    expo = exp(expo)
    f = mul(qt, expo)

    return f


def loss_motion(qt: np.array, qt_bfr: np.array, omega: np.array, interval: np.float32) -> jnp.float32:
    """
    Return the motion model loss, which is the loss between the motion model
    and the real quaternion

    Args:
        qt: quaternion inside the motion model
        qt_bfr: qiaternion at the previous moment just before qt
        omega: the anguler velocity at the moment as the same as qt
        interval: time interval between qt and qt_n

    Returns:
        loss: the loss between the motion model and the real quaternion
    """
    f = motion(qt_bfr, omega, interval)
    log_term = 2*log(mul(inv(qt), f))
    loss = jnp.square(jnp.linalg.norm(log_term)) * 0.5

    return loss


def observ(qt: np.array) -> jnp.array:
    """
    Calculate the observation model, which is the observation of gravity
    in IMU frame.

    Args:
        qt: quaternion at the moment t

    Returns:
        h: obeservation model.
    """
    g = np.array([0, 0, 0, -1])
    h = mul(mul(inv(qt), g), qt)

    return h


def loss_observ(qt: np.array, at: np.array) -> jnp.array:
    """
    Calculate the loss between observation model and the measurements

    Args:
        qt: quaternion at the moment t
        at: accelerometer measurements at the moment t

    Returns:
        loss: the loss between observation model and the measurements
    """
    at = jnp.concatenate([jnp.array([0]), at])
    ht = observ(qt)
    loss = jnp.square(jnp.linalg.norm(at-ht)) * 0.5

    return loss


@jit
def get_cost(qT: np.array, imuData: np.array, ts: np.array) -> jnp.float32:
    """
    Calculate the cost function of the estimated quaternions and measurements
    IMU data.

    Args:
        qT: the quaternion matrix, with the shape of 4 x batchsize, pay attention qT
            does not include q0
        imuData: the measurements
        ts: the time series corresponding to the imu data

    Returns:
        cost: the final cost
    """
    T = ts.size - 1
    qT = jnp.asarray(qT)
    imuData = jnp.asarray(imuData)
    ts = jnp.asarray(ts)

    q0 = jnp.array([1, 0, 0, 0])[:, None]
    q_all = jnp.concatenate([q0, qT], axis=1)

    intervals = ts[0, 1:] - ts[0, :-1]
    intervals = intervals[None, :]

    # Vectorized calculation
    v_loss_motion = jax.vmap(loss_motion, 1, 0)
    v_loss_observ = jax.vmap(loss_observ, 1, 0)
    
    cost_motion = jnp.sum(v_loss_motion(q_all[:, 1:], q_all[:, :-1], imuData[3:, 1:], intervals))
    cost_observ = jnp.sum(v_loss_observ(qT, imuData[:3, 1:]))

    cost = cost_motion + cost_observ


    return cost


cost_descent = jax.jacrev(get_cost)


@jit
def projection(gk: jnp.array, xk: jnp.array) -> jnp.array:
    """
    Project the gradient to the tangent plane in order to obey the constraint that quaternions need
    to be uniform

    Args:
        gk: gradient at the k step
        xk: vector at the k step

    Returns:
        nk: normalized projected gradient at the k step
    """
    nk = gk - gk.dot(xk)*xk
    nk = nk/jnp.linalg.norm(nk)

    return nk


v_projection = jax.vmap(projection, 1, 1) # vectorized projection


def init_qT(omega: np.array, ts: np.array) -> np.array:
    """
    Initialize the quaternion matrix using motion model assuming that the 
    initial state is [1, 0, 0, 0]

    Args:
        omega: the measurements of the anguler velocity
        ts: time sequence

    Returns:
        qT: the quaternion array that need to be optimized (the initial state is not included)
    """
    qT = np.array([1,0,0,0])[:, None]
    T = ts.size - 1
    
    for i in range(T):
        qt = motion(qT[:, -1], omega[:, i+1], ts[0, i+1] - ts[0, i])[:, None]
        qT = np.concatenate([qT, qt], axis=1)

    return qT[:, 1:]
    