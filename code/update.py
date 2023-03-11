from pr3_utils import *
from constants import *
from numba_helper import *


@njit((nb.f8[:, ::1], nb.f8[:, :, ::1], nb.f8[:, ::1], nb.f8[:, ::1],
                         nb.f8[:, ::1], nb.f8[:, ::1], nb.f8, nb.f8[:, ::1]))
def update_lmk(mu_lmk: np.ndarray, Sigma_lmk: np.ndarray, feature: np.ndarray, pose: np.ndarray, 
               Ks: np.ndarray, K: np.ndarray, b: np.float64, imu_T_cam: np.ndarray):
    """
    This function updates the locations of the landmarks.

    Args:
        mu_lmk: the homogeneous world frame point coordinates of the landmarks of the previous state. Shape [#feature, 4]
        Sigma_lmk: the covariance matrix of world frame point coordinates of the landmarks. Shape [# feature, 3, 3]
        feature: the observed features at current state. Shape [#feature, 4] 
            (the pixel coordinates [ul vl ur vr] of the feature)
        pose: the imu pose world_T_imu. Shape [4, 4]
        Ks: the stereo calibration matrix. Shape [4, 4]
        K: the single camera calibration matrix. Shape [4, 4]
        cam_T_imu: the pose from imu frame to cam frame. Shape [4, 4]

    Returns:
        mu_lmk_next: updated poses of the landmarks.
        Sigma_lmk_next: updated sigmas of the landmarks
    """
    world_T_cam = pose @ imu_T_cam
    cam_T_world = np.linalg.inv(world_T_cam)
    n_feature = feature.shape[0]
    mu_lmk_next = mu_lmk.copy()
    Sigma_lmk_next = Sigma_lmk.copy()

    for i in range(n_feature):
        zi = feature[i]

        # If the feature is not observable, skip it
        if (zi == -1).all():
            Sigma_lmk_next[i] = np.eye(3) * 0.01
            continue
            # pass

        # If the feature is firstly observed
        if np.isnan(mu_lmk[i]).any():
            disparity = zi[0] - zi[2]
            z0 = K[0, 0] * b / disparity
            mu_lmk_next[i] = world_T_cam @ \
                        np.concatenate(
                            (z0 * np.linalg.inv(K) @ np.concatenate((zi[:2], np.array([1]))), 
                                np.array([1]))
                            )
            Sigma_lmk_next[i] = np.eye(3) * 0.01
            continue

        # Else calculate the predicted feature
        lmk_coordinates_cam = cam_T_world @ mu_lmk[i]
        zi_tilde = Ks @ pi(lmk_coordinates_cam)
        # print(zi_tilde)

        # calculate the Jacobian matrix of the observation model
        H = Ks @ dpi_dq(lmk_coordinates_cam) @ cam_T_world @ P.T

        # Calculate the kalman gain
        v_diag = np.random.normal(loc=0, scale=sigma_v, size=(4))
        V = np.diag(v_diag ** 2)
        Kg = Sigma_lmk[i] @ H.T @ np.linalg.inv(H @ Sigma_lmk[i] @ H.T + V)
        mu_lmk_next[i] = mu_lmk[i] + P.T @ Kg @ (zi - zi_tilde)
        Sigma_lmk_next[i] = (np.eye(3) - Kg @ H) @ Sigma_lmk[i]

    return mu_lmk_next, Sigma_lmk_next

@njit((nb.f8[:, ::1], nb.f8[:, ::1], nb.f8[:, ::1], nb.f8[:, ::1],
                nb.f8[:, ::1], nb.f8[:, ::1], nb.f8, nb.f8[:, ::1]))
def update_lmk_ekf(mu_lmk: np.ndarray, Sigma_lmk: np.ndarray, feature: np.ndarray, pose: np.ndarray, 
               Ks: np.ndarray, K: np.ndarray, b: np.float64, imu_T_cam: np.ndarray):
    """
    This function updates the locations of the landmarks.

    Args:
        mu_lmk: the homogeneous world frame point coordinates of the landmarks of the previous state. Shape [#feature, 4]
        Sigma_lmk: the covariance matrix of world frame point coordinates of the landmarks. Shape [3#feature, 3#feature]
        feature: the observed features at current state. Shape [#feature, 4] 
            (the pixel coordinates [ul vl ur vr] of the feature)
        pose: the imu pose world_T_imu. Shape [4, 4]
        Ks: the stereo calibration matrix. Shape [4, 4]
        K: the single camera calibration matrix. Shape [4, 4]
        cam_T_imu: the pose from imu frame to cam frame. Shape [4, 4]

    Returns:
        mu_lmk_next: updated poses of the landmarks.
        Sigma_lmk_next: updated sigmas of the landmarks
    """
    mu_lmk_next = mu_lmk.copy()
    Sigma_lmk_next = Sigma_lmk.copy()
    # print(mu_lmk_next.shape)
    world_T_cam = pose @ imu_T_cam
    cam_T_world = np.linalg.inv(world_T_cam)
    

    # index_observed = np.all(feature != -1, axis=1)
    # (feature != -1).all(axis=1) # find the indexes that indexes are observed
    # print(np.where(index_observed==True)[0].shape)
    # index_notnan = ~np.isnan(mu_lmk).any(axis=1)
    # index_update = np.logical_and(index_observed, index_notnan)
    # index_update = np.where((~np.isnan(mu_lmk).all(axis=1)) & (feature != -1).all(axis=1))[0]
    index_update = np.where((~np.isnan(mu_lmk[:, 0])) & (feature[:, 0] != -1))[0]
    # print(index_update.shape)


    # index_initialize = np.logical_and(index_observed, np.isnan(mu_lmk).any(axis=1))
    # index_initialize = np.where((feature != -1) & np.isnan(mu_lmk))[0]
    index_initialize = np.where((feature[:, 0] != -1) & np.isnan(mu_lmk[:, 0]))[0]
    # index_initialize = np.where(index_initialize)[0]
    mu_lmk_initialize = mu_lmk[index_initialize]

    feature_initialize = feature[index_initialize]
    
    # Initialize the observed but with-no-previous-state points 
    for i in range(mu_lmk_initialize.shape[0]):
        zi = feature_initialize[i]
        disparity = zi[0] - zi[2]
        z0 = K[0, 0] * b / disparity
        mu_lmk_initialize[i] = world_T_cam @ \
                    np.concatenate(
                        (z0 * np.linalg.inv(K) @ np.concatenate((zi[:2], np.array([1]))), 
                            np.array([1]))
                        )
        j = index_initialize[i]
        Sigma_lmk_next[j:j+3, j:j+3] = np.eye(3) * 0.01
        continue

    mu_lmk_next[index_initialize] = mu_lmk_initialize


    # print(np.where(index_initialize==True)[0].shape)
    z_cur = feature[index_update].reshape((-1, 1))
    # Nt = np.where(index_update==True)[0].size
    Nt = index_update.size

    M = feature.shape[0]
    # print(Nt, index_valid)
    Ht = np.zeros((4*Nt, 3*M), dtype=np.float64)
    z_tilde = np.zeros((4*Nt, 1), dtype=np.float64)
    for i in range(Nt):
        # j = index_valid[i]
        j = index_update[i]
        # calculate the z tilde
        lmk_coordinates_cam = cam_T_world @ mu_lmk[j]
        zi_tilde = Ks @ pi(lmk_coordinates_cam)
        z_tilde[i:i+4, 0] = zi_tilde
        # calculate the Jacobian matrix of the observation model
        Ht[i:i+4, j:j+3] = Ks @ dpi_dq(lmk_coordinates_cam) @ cam_T_world @ P.T

    if Nt != 0:
        v_diag = np.random.normal(loc=0, scale=sigma_v, size=(4*Nt))
        V = np.diag(v_diag ** 2)

        # print(Ht.shape, Sigma_lmk_next.shape, V.shape)
        Kt = Sigma_lmk_next @ Ht.T @ np.linalg.inv(Ht @ Sigma_lmk_next @ Ht.T + V)
        mu_t = np.ascontiguousarray(mu_lmk_next.copy()[:, :3])
        mu_t = mu_t.reshape((-1, 1))
        
        # print('mu_t', mu_t.shape, Kt.shape, z_cur.shape, z_tilde.shape)
        mu_next = mu_t + Kt @ (z_cur - z_tilde)
        
        mu_lmk_next[:, :3] = mu_next.reshape((-1, 3)) # renew the mean of landmarks
        Sigma_lmk_next = (np.eye(3*M) - Kt @ Ht) @ Sigma_lmk_next


    return mu_lmk_next, Sigma_lmk_next