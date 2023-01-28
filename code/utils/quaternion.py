import jax.numpy as jnp
import jax


def conj(p: jnp.array) -> jnp.array:
    """
    Return the conjuagate of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the conjugate
    """
    return jnp.concatenate([p[0][None], -p[1:]])


def norm(p: jnp.array) -> jnp.array:
    """
    Return the norm of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: norm of the quaternion
    """
    return jnp.sqrt(p[0]**2 + p[1:].dot(p[1:]))


def inv(p: jnp.array) -> jnp.array:
    """
    Return the inverse of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the inverse of the quaternion
    """
    return conj(p)/(norm(p)**2)


def mul(q: jnp.array, p: jnp.array) -> jnp.array:
    """
    Calculate the multiplication of two quaternions q and p.

    Args:
        p: quaternion, represented by jnp array
        q: quaternion, represented by jnp array

    Returns:
        result: multiplied quarternion
    """
    # result = jnp.zeros(4)
    pre = q[0]*p[0] - q[1:].dot(p[1:])[None]
    suf = q[0]*p[1:]+p[0]*q[1:] + jnp.cross(q[1:], p[1:])
    result = jnp.concatenate([pre, suf])

    return result


def exp(p: jnp.array) -> jnp.array:
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


def log(p: jnp.array) -> jnp.array:
    """
    Calculate the logarithm of the quaternion

    Args:
        p: quaternion, represented by jnp array

    Returns:
        val: the logarithm of the quaternion
    """
    p_norm = norm(p)
    pv = p[1:]
    pv_norm = jnp.sqrt(pv.dot(pv))
    val = jnp.concatenate(
        [jnp.log(p_norm)[None], pv/pv_norm*jnp.arccos(p[0]/p_norm)])
    return val


def mulP(params, shift):
    a, b = params
    return mul(a, b)*shift

if __name__ == "__main__":
    a = jnp.ones(4)
    b = jnp.ones(4)*2

    print(norm(mul(a, b)), norm(a)*norm(b))

    print(exp(a))
    gr = jax.jacrev(mul)
    print(gr(a, b))

    params = [a, b]

    grr = jax.jacrev(mulP)
    print(grr(params, 2))


    # print(mul())
