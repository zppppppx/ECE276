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
    return jnp.sqrt(p.dot(p))


def inv(p: np.array) -> np.array:
    """
    Return the inverse of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the inverse of the quaternion
    """
    # p += np.array([0, 1e-3, 1e-3, 1e-3])
    return conj(p)/((norm(p)+1e-3)**2)


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
    # q += np.array([0, 1e-3, 1e-3, 1e-3], dtype=np.float32)
    # p += np.array([0, 1e-3, 1e-3, 1e-3], dtype=np.float32)
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
    # p += np.array([0, 1e-3, 1e-3, 1e-3], dtype=np.float32)
    pv = p[1:]
    pv_norm = jnp.sqrt(pv.dot(pv)) + 1e-3
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
    # p += np.array([0, 1e-3, 1e-3, 1e-3], dtype=np.float32)
    p_norm = norm(p)
    pv = p[1:]
    pv_norm = jnp.sqrt(pv.dot(pv)) + 1e-3

    return jnp.where(jnp.isclose(pv_norm, 0), jnp.array([jnp.log(p_norm), 0, 0, 0]), 
        jnp.concatenate([jnp.log(p_norm)[None], pv/pv_norm*jnp.arccos(p[0]/p_norm)]))


def gen_quaternion(len) -> np.array:
    uniform = lambda x : x/np.sqrt(x.dot(x))
    quaternions = np.random.rand(len, 4)
    # quaternions = uniform(quaternions)
    quaternions = np.array([*map(uniform, quaternions)])
    # quaternions[0, :] = np.array([1,0,0,0])

    return quaternions.T

v_mul = jax.vmap(mul, 1, 1)
v_log = jax.vmap(log, 1, 1)
v_inv = jax.vmap(inv, 1, 1)
v_norm = jax.vmap(norm, 1, 0)

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


    # for i in range(1000):
    #     qt = np.random.rand(4)
    #     qt = qt/np.linalg.norm(qt)
    #     gr = jax.jacrev(log)
    #     grad = gr(qt)
    #     if (jnp.isnan(grad.any())):
    #         print("yes")

    qt = np.array([1.,0.,0.,0.])
    gr = jax.jacrev(conj)
    grad = gr(qt)
    print(grad)
