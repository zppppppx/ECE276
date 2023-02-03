import jax.numpy as jnp
import jax
import numpy as np
import transforms3d.quaternions as tq

def conj(p: np.array) -> np.array:
    """
    Return the conjuagate of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the conjugate
    """
    return jnp.concatenate([p[0][None], -p[1:]])


def norm(p: np.array) -> np.array:
    """
    Return the norm of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: norm of the quaternion
    """
    return jnp.sqrt(p[0]**2 + p[1:].dot(p[1:]))


def inv(p: np.array) -> np.array:
    """
    Return the inverse of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the inverse of the quaternion
    """
    return conj(p)/(norm(p)**2)


def mul(q: np.array, p: np.array) -> np.array:
    """
    Calculate the multiplication of two quaternions q and p.

    Args:
        p: quaternion, represented by jnp array
        q: quaternion, represented by jnp array

    Returns:
        result: multiplied quarternion
    """
    # result = jnp.zeros(4)
    # print(q, p)
    pre = q[0]*p[0] - q[1:].dot(p[1:])[None]
    suf = q[0]*p[1:]+p[0]*q[1:] + jnp.cross(q[1:], p[1:])
    result = jnp.concatenate([pre, suf])
    # print(result)

    return result


def exp(p: np.array) -> np.array:
    """
    Calculate the exponential of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the exponential of the quaternion
    """
    pv = p[1:]
    pv_norm = jnp.sqrt(pv.dot(pv))
    val = jnp.concatenate(
        [jnp.cos(pv_norm)[None], pv/pv_norm*jnp.sin(pv_norm)])
    return jnp.exp(p[0])*val


def log(p: np.array) -> np.array:
    """
    Calculate the logarithm of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the logarithm of the quaternion
    """
    # print(p)
    p_norm = norm(p)
    pv = p[1:]
    pv_norm = jnp.sqrt(pv.dot(pv)) + 1e-7
    # if(pv_norm <= 1e-7):
    val = jnp.concatenate(
        [jnp.log(p_norm)[None], pv/pv_norm*jnp.arccos(p[0]/p_norm)])
    # else:
    #     val = jnp.asarray([jnp.log(p[0]), 0, 0, 0])
    # print(p_norm, pv_norm)
    return val


def gen_quaternion(len) -> np.array:
    uniform = lambda x : x/np.sqrt(x.dot(x))
    quaternions = np.random.rand(len, 4)
    # quaternions = uniform(quaternions)
    quaternions = np.array([*map(uniform, quaternions)])
    # quaternions[0, :] = np.array([1,0,0,0])

    return quaternions.T

if __name__ == "__main__":
    qt = gen_quaternion(1).squeeze()
    print(qt)
    print("norm:", np.allclose(norm(qt), tq.qnorm(qt)))
    print("inv:", np.allclose(inv(qt), tq.qinverse(qt)))
    print("conjugate:", np.allclose(conj(qt), tq.qconjugate(qt)))
    print("log:", np.allclose(log(qt), tq.qlog(qt)))
    print("exp:", np.allclose(exp(qt), tq.qexp(qt)))
    print("mul(qt, inv(qt):", np.allclose(mul(qt, inv(qt)), tq.qmult(qt, tq.qinverse(qt))))

    print("Normal norm: ", np.linalg.norm(np.array([1,2,3,4])))
