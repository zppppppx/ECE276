import numpy as np
import transforms3d as t3d
import jax
import jax.numpy as jnp

class Config: 
    Vref = 3300
    sensitivity_gyro = 4
    sensitivity_accer = 300
    max_index = 1023
    scale_factor_gyro = Vref/max_index/sensitivity_gyro
    scale_factor_accer = Vref/max_index/sensitivity_accer


def rot2euler(rot: np.array, axes: str = 'sxyz'):
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
    T = rot.shape[-1]
    x, y, z = np.array([]), np.array([]), np.array([])
    for i in range(T):
        nx, ny, nz = t3d.euler.mat2euler(rot[:, :, i], axes=axes)
        x = np.concatenate([x, np.asarray([180 * nx / np.pi])])
        y = np.concatenate([y, np.asarray([180 * ny / np.pi])])
        z = np.concatenate([z, np.asarray([180 * nz / np.pi])])

    return x, y, z

    

def findBias(eulers: list, imuData: np.array, threshold: np.float32) -> np.array:
    """
    Return the static frames according to rotation matrix and use them to find the bias from
    imuData. Threshold is the extent to decide the rotation.

    Args:
        eulers: euler angles for x, y and z
        imuData: imu measurements
        threshold: the extent needed to decide whether the rotation has begun.

    Returns:
        bias: 6 x batch sized bias to calibrate the imu data
    """
    # In-place replace
    imuData[:2] *= -1 # reverse the ax and ay to normal representation
    imuData[3:] = imuData[[5,4,3]] # reverse the anguer velocity to normal order

    x, y, z = eulers
    # the frame lists that satisfy the threshold. 
    x_start, y_start, z_start = np.argwhere(np.abs(x-x[0]) >= threshold), \
        np.argwhere(np.abs(y-y[0]) >= threshold), np.argwhere(np.abs(z-z[0]) >= threshold)
    end_frame = np.min([x_start[0], y_start[0], z_start[0]]) # the frame that static state ends
    start_frame = end_frame - 200 if end_frame >= 200 else 0

    bias = imuData[:, start_frame:end_frame]
    bias = np.average(bias, axis=1)

    return bias


def calibrate(imuData: np.array, bias: np.array) -> np.array:
    """
    Calibrate the imu data according to fetched bias

    Args:
        imuData: the imu data waiting to be calibrated
        bias: fetched bias

    Return:
        result: calibrated data
    """
    bias[2] -= 1. / Config.scale_factor_accer
    print(bias)
    imuData -= bias[:, None]

    imuData[:3] *= Config.scale_factor_accer
    imuData[3:] *= Config.scale_factor_gyro * np.pi / 180


    return imuData

