import numpy as np
from utils.calibrate import *
import sys
sys.path.append("..")


class Config:
    theta_min = 3*np.pi/8  # 3/8 pi
    theta_rg = np.pi/4

    phi_max = np.pi/6
    phi_rg = np.pi/3


def genPanorama(cam_data: np.array, ts_cam: np.array, ts_imu: np.array, qT: np.array):
    """
    Generate the panorama

    Args:
        cam_data: the pictures taken by camera, with shape [r x c x ch x batch]
        ts_cam: the time stamp of when each picture was taken
        ts_imu: the time stamp of when each IMU measurement was obtained
        qT: the quaternion matrixes

    Returns:
        panorama: the panorama picture
    """
    n_rows, n_cols, n_chanls, n_pics = cam_data.shape
    n_rows_p, n_cols_p = n_rows*4, n_cols*6 # to fit in with the pi*2pi

    imu_ts_index = 0
    T = ts_imu.shape[-1]

    qT = np.concatenate([np.array([1., 0., 0., 0.])[:, None], qT], axis=1)
    rots = quat2rot(qT)

    cartesian = genCartesian(n_rows, n_cols)
    panorama = np.zeros([n_rows_p, n_cols_p, n_chanls], dtype=cam_data.dtype)

    for i in range(n_pics):
        cam_time = ts_cam[0, i]
        # loop until the imu timestamp is just next to the camera timestamp

        while(imu_ts_index + 1 < T and ts_imu[0, imu_ts_index+1] <= cam_time):
            imu_ts_index += 1

        rot = rots[:, :, imu_ts_index]
        coor_world = rot.dot(cartesian) + np.array([0,0,0.1])[:, None]

        coor_world /= np.linalg.norm(coor_world, axis=0) # unit sphere

        # Convert to cylinder
        coor_world[0, :] = (0.5 * (n_cols_p - 1) / np.pi) * (np.pi + np.arctan2(coor_world[1, :], coor_world[0, :]))

        coor_world[1, :] = ((n_rows_p - 1) / np.pi) * np.arccos(coor_world[2, :])

        panorama[np.round(coor_world[1, :]).astype(np.int32), np.round(coor_world[0, :]).astype(np.int32)] \
            = cam_data[:, :, :, i].reshape([n_rows * n_cols, n_chanls])
        
    return panorama

        
def genCartesian(n_rows, n_cols):
    """
    Generate the Cartesian coordianates for one picture with n_rows and n_cols under camera frame

    Args:
        n_rows: the number of rows of the pixels in the picture
        n_cols: the number of columns of the pixels in the picture

    Returns:
        cartesian: the cartesian coordinates of each pixel in the picture. Reshaped to [3 x n_rows*n_cols]
            to accelerate the calculation
    """
    # the cartesian coordinates at pixels
    cartesian = np.ones([n_rows, n_cols, 3])

    for i in range(n_rows):
        theta = Config.theta_min + i * Config.theta_rg / (n_rows-1)
        sin_theta = np.sin(theta)
        cartesian[i, :, :] = [sin_theta, sin_theta, np.cos(theta)]

    for i in range(n_cols):
        phi = Config.phi_max - i * Config.phi_rg / (n_cols-1)
        cartesian[:, i, 0] *= np.cos(phi)
        cartesian[:, i, 1] *= np.sin(phi)

    return cartesian.reshape([n_rows * n_cols, 3]).T
