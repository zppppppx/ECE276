import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def motion(qt: np.array, omega: np.array, interval: np.float32) -> np.array:
    """
    Calculate the quaternion kinematics motion model

    Args:
        qt: quaternion, represented by np array
        omega: the anguler velocity detected by IMU
        interval: the interval between this measurement and the last measurement

    Returns:
        f: calculated motion model, which is q_(t+1)
    """
    expo = np.concatenate([np.array([0]), interval*omega/2])
    expo = exp(expo)
    f = mul(qt, expo)

    return f


def mul(q: np.array, p: np.array) -> np.array:
    """
    Calculate the multiplication of two quaternions q and p.

    Args:
        p: quaternion, represented by np array
        q: quaternion, represented by np array

    Returns:
        result: multiplied quarternion
    """
    pre = q[0]*p[0] - q[1:].dot(p[1:])[None]
    suf = q[0]*p[1:]+p[0]*q[1:] + np.cross(q[1:], p[1:])
    result = np.concatenate([pre, suf])

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
    pv_norm = np.sqrt(pv.dot(pv))
    val = np.concatenate(
        [np.cos(pv_norm)[None], pv/pv_norm*np.sin(pv_norm)])
    return np.exp(p[0])*val


def quat2rot(qT: np.array):
    """
    Find the corresponding three euler angles according to the rotation matrix

    Args:
        rot: rotation matrix
        axes: rotation matrix transformation method

    Returns
        x: the angle of roll
        y: the angle of pitch
        z: the angle of yaw
    """
    T = qT.shape[-1]

    for i in range(T):
        rot = t3d.quaternions.quat2mat(qT[:, i])
        if(i == 0):
            rots = rot[:, :, None]
        else:
            rots = np.concatenate([rots, rot[:, :, None]], axis=2)

    return rots

def plot_trajectory(slam, n_particles: int, noised: bool=True):
    """
    Plot the trajectory of the particles
    """
    print("######################### Plotting the trajectory ########################")
    print
    for i in range(n_particles):
        pos = slam.get_trajectory(noised=noised)
        plt.plot(pos[0, :], pos[1, :])
        
    plt.legend(list(range(n_particles)))
    plt.show()


def plot_particle_trajectory(slam):
    """
    Plot the predict-only trajectory using particle calculation
    """
    trajectories = slam.predict_only()
    n = trajectories.shape[0]
    for i in range(n):
        plt.plot(trajectories[i, 0], trajectories[i, 1])

    plt.legend(list(range(n)))
    plt.show()


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # print(cutoff, fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    print(cutoff, fs)
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


