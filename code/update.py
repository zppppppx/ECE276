from pr3_utils import *
from constants import *
from numba_helper import *
from numpy.linalg import inv
from scipy.linalg import expm

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
    cam_T_world = inv(world_T_cam)
    n_feature = feature.shape[0]
    mu_lmk_next = mu_lmk.copy()
    Sigma_lmk_next = Sigma_lmk.copy()

    for i in range(n_feature):
        zi = feature[i]

        # If the feature is not observable, skip it
        if (zi == -1).all():
            Sigma_lmk_next[i] = np.eye(3) * sigma_lmk
            continue
            # pass

        # If the feature is firstly observed
        if np.isnan(mu_lmk[i]).any():
            disparity = zi[0] - zi[2]
            z0 = K[0, 0] * b / disparity
            mu_lmk_next[i] = world_T_cam @ \
                        np.concatenate(
                            (z0 * inv(K) @ np.concatenate((zi[:2], np.array([1]))), 
                                np.array([1]))
                            )
            Sigma_lmk_next[i] = np.eye(3) * sigma_lmk
            continue

        # Else calculate the predicted feature
        lmk_coordinates_cam = cam_T_world @ mu_lmk[i]
        zi_tilde = Ks @ pi(lmk_coordinates_cam)
        # print(zi_tilde)

        # calculate the Jacobian matrix of the observation model
        H = Ks @ dpi_dq(lmk_coordinates_cam) @ cam_T_world @ (P.T)

        # Calculate the kalman gain
        # v_diag = np.random.normal(loc=0, scale=sigma_v, size=(4))
        # V = np.diag(v_diag ** 2)
        Kg = Sigma_lmk[i] @ H.T @ inv(H @ Sigma_lmk[i] @ H.T + V)
        # print(Kg.shape)
        mu_lmk_next[i] = mu_lmk[i] + (P.T) @ Kg @ (zi - zi_tilde)
        Sigma_lmk_next[i] = (np.eye(3) - (Kg @ H)) @ Sigma_lmk[i]

    return mu_lmk_next, Sigma_lmk_next


# @njit((nb.f8[:, ::1], nb.f8[:, ::1], nb.f8[:, ::1], nb.f8[:, ::1],
#                 nb.f8[:, ::1], nb.f8[:, ::1], nb.f8, nb.f8[:, ::1]))
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
    cam_T_world = inv(world_T_cam)
    
    index_update = np.where((~np.isnan(mu_lmk[:, 0])) & (feature[:, 0] != -1))[0]
    index_initialize = np.where(np.isnan(mu_lmk[:, 0]) & (feature[:, 0] != -1))[0]

    mu_lmk_initialize = mu_lmk[index_initialize]
    feature_initialize = feature[index_initialize]
    
    # Initialize the observed but with-no-previous-state points 
    for i in range(mu_lmk_initialize.shape[0]):
        zi = feature_initialize[i]
        disparity = zi[0] - zi[2]
        z0 = K[0, 0] * b / disparity
        mu_lmk_initialize[i] = world_T_cam @ \
                    np.concatenate(
                        (z0 * inv(K) @ np.concatenate((zi[:2], np.array([1]))), 
                            np.array([1]))
                        )
        j = index_initialize[i]
        Sigma_lmk_next[3*j:3*j+3, 3*j:3*j+3] = np.eye(3) * 2
        continue

    mu_lmk_next[index_initialize] = mu_lmk_initialize


    # print(np.where(index_initialize==True)[0].shape)
    z = feature[index_update].reshape((-1, 1))
    # Nt = np.where(index_update==True)[0].size
    Nt = index_update.size
    if Nt == 0:
        return mu_lmk_next, Sigma_lmk_next

    M = feature.shape[0]
    # print(Nt, index_valid)
    Ht = np.zeros((4*Nt, 3*M), dtype=np.float64)
    z_tilde = np.zeros((4*Nt, 1), dtype=np.float64)
    for i in range(Nt):
        # j = index_valid[i]
        j = index_update[i]
        # calculate the z tilde
        lmk_coordinates_cam = cam_T_world @ mu_lmk[j]
        # zi_tilde = Ks @ pi(lmk_coordinates_cam)
        zi_tilde = Ks @ projection(lmk_coordinates_cam)
        z_tilde[4*i:4*i+4, 0] = zi_tilde
        # calculate the Jacobian matrix of the observation model
        # Ht[4*i:4*i+4, 3*j:3*j+3] = Ks @ dpi_dq(lmk_coordinates_cam) @ cam_T_world @ (P.T)
        Ht[4*i:4*i+4, 3*j:3*j+3] = Ks @ projectionJacobian(lmk_coordinates_cam) @ cam_T_world @ (P.T)

    # a = np.mean(np.abs(z-z_tilde))
    # print(a)
    # print(z[:4], z_tilde[:4])
    # print(np.isnan(z_tilde).any(), np.isnan(z).any())

    if Nt != 0:
        # v_diag = np.random.normal(loc=0, scale=sigma_v, size=(4*Nt))
        # V = np.diag(v_diag ** 2)

        # print(Ht.shape, Sigma_lmk_next.shape, V.shape)
        Kt = Sigma_lmk @ (Ht.T) @ inv(Ht @ Sigma_lmk @ (Ht.T) + np.kron(np.eye(Nt), V))
        # mu_t = np.ascontiguousarray(mu_lmk_next.copy()[:, :3])
        mu_t = mu_lmk_next[:, :3].copy().reshape((-1, 1))
        
        # print('mu_t', mu_t.shape, Kt.shape, z_cur.shape, z_tilde.shape)
        mu_next = mu_t + Kt @ (z - z_tilde)

        # print((Kt @ (z - z_tilde)).shape)
        
        mu_lmk_next[index_update, :3] = mu_next.reshape((-1, 3))[index_update] # renew the mean of landmarks
        Sigma_next = (np.eye(3*M) - (Kt @ Ht)) @ Sigma_lmk_next
        indexes_update = 3 * index_update
        indexes_update = np.concatenate((indexes_update, indexes_update+1, indexes_update+2))
        Sigma_lmk_next[indexes_update[:, np.newaxis], indexes_update] = Sigma_next[indexes_update[:, np.newaxis], indexes_update]


    return mu_lmk_next, Sigma_lmk_next


def update_pose_ekf(mu_predicted: np.ndarray, Sigma_pose: np.ndarray, feature: np.ndarray, mu_lmk: np.ndarray,
               Ks: np.ndarray, K: np.ndarray, b: np.float64, imu_T_cam: np.ndarray):
    """
    This function updates the pose of the robot.

    Args:
        mu_predicted: the predicted pose of the robot. Shape [4, 4]
        Sigma_pose: the covariance of the pose. NOTE that it is not the covariance of the pose matrix but the 
            covariance of the general form of the pose: [pos^T, angle^T]^T. Shape [6, 6]
        feature: the observed features at current state. Shape [#feature, 4] 
            (the pixel coordinates [ul vl ur vr] of the feature)
        mu_lmk: the mean positions of the features before its updation (the previous state). Shape [#feature, 4]
        Ks: the stereo calibration matrix. Shape [4, 4]
        K: the single camera calibration matrix. Shape [4, 4]
        cam_T_imu: the pose from imu frame to cam frame. Shape [4, 4]

    Returns:
        mu_updated: updated poses of the landmarks.
        Sigma_updated: updated sigmas of the landmarks
    """
    world_T_cam = mu_predicted @ imu_T_cam
    cam_T_world = inv(world_T_cam)
    cam_T_imu = inv(imu_T_cam)
    mu_updated = mu_predicted.copy()
    Sigma_pose_updated = Sigma_pose.copy()
    # first find the valid indexes that the features are observed and the positions have previously been intialized
    index_update = np.where((~np.isnan(mu_lmk[:, 0])) & (feature[:, 0] != -1))[0]
    Nt = index_update.size

    # if there are no valid observations, use the prediected pose as the updated one
    if Nt == 0:
        return mu_updated, Sigma_pose_updated
    
    # Ht = []
    # z_tilde = []
    Ht = np.zeros((4*Nt, 6))
    z_tilde = np.zeros((4*Nt))
    z = feature[index_update].reshape((-1))
    # print(z.shape, Nt)
    for i in range(Nt):
        j = index_update[i]
        lmk_coordinates_cam = cam_T_world @ mu_lmk[j]
        zi_tilde = Ks @ pi(lmk_coordinates_cam)
        # z_tilde.append(zi_tilde)
        z_tilde[4*i:4*i+4] = zi_tilde
        Hti = -Ks @ dpi_dq(lmk_coordinates_cam) @ cam_T_imu @ cdot_hat(inv(mu_predicted) @ mu_lmk[j])
        # Ht.append(Hti)
        Ht[4*i:4*i+4] = Hti

    

    # v_diag = np.random.normal(loc=0, scale=sigma_v, size=(4))
    # V = np.diag(v_diag ** 2)
    Kt = Sigma_pose @ (Ht.T) @ inv(Ht @ Sigma_pose @ (Ht.T) + np.kron(np.eye(Nt), V))
    mu_updated = mu_predicted @ expm(angle2twist(Kt @ (z - z_tilde)))
    Sigma_pose_updated = (np.eye(6) - (Kt @ Ht)) @ Sigma_pose

    return mu_updated, Sigma_pose_updated

    # Ht = np.vstack(tuple(Ht))
    # z_tilde = np.vstack(tuple(z_tilde))


# @njit((nb.f8[:, ::1], nb.f8[:, ::1], nb.f8[:, ::1], nb.f8[:, ::1],
#                 nb.f8[:, ::1], nb.f8[:, ::1], nb.f8, nb.f8[:, ::1]))
def update_slam_ekf(mu_lmk: np.ndarray, Sigma: np.ndarray, feature: np.ndarray, pose: np.ndarray, 
               Ks: np.ndarray, K: np.ndarray, b: np.float64, imu_T_cam: np.ndarray):
    """
    This function updates the pose and the mean of the features together.

    Args:
        mu_lmk: the homogeneous world frame point coordinates of the landmarks of the previous state. Shape [#feature, 4]
        Sigma: the covariance matrix of the robot pose and landmark means. Shape [6 + 3#feature, 6 + 3#feature]
        feature: the observed features at current state. Shape [#feature, 4] 
            (the pixel coordinates [ul vl ur vr] of the feature)
        pose: the imu pose world_T_imu (predicted pose from the previous step). Shape [4, 4]
        Ks: the stereo calibration matrix. Shape [4, 4]
        K: the single camera calibration matrix. Shape [4, 4]
        cam_T_imu: the pose from imu frame to cam frame. Shape [4, 4]

    Returns:
        pose_updated
        mu_lmk_updated
        Sigma_updated
    """
    pose_updated = pose.copy()
    mu_lmk_updated = mu_lmk.copy()
    Sigma_updated = Sigma.copy()

    # print(mu_lmk_next.shape)
    world_T_cam = pose @ imu_T_cam
    cam_T_world = inversePose(world_T_cam)
    cam_T_imu = inversePose(imu_T_cam)


    index_update = np.where((~np.isnan(mu_lmk[:, 0])) & (feature[:, 0] != -1))[0]
    # print(index_update.shape)

    index_initialize = np.where(np.isnan(mu_lmk[:, 0]) & (feature[:, 0] != -1))[0]
    mu_lmk_initialize = mu_lmk[index_initialize]

    feature_initialize = feature[index_initialize]
    distance = []
    for i in range(mu_lmk_initialize.shape[0]):
        zi = feature_initialize[i]
        disparity = zi[0] - zi[2]
        z0 = K[0, 0] * b / disparity
        lmk_coordinates_cam = np.concatenate(
                        (z0 * inv(K) @ np.concatenate((zi[:2], np.array([1]))), np.array([1])))
        distance.append(np.sqrt(np.sum(lmk_coordinates_cam**2-1)))
        mu_lmk_initialize[i] = world_T_cam @ lmk_coordinates_cam

    if(len(distance) != 0):
        print("Initialization max and minimum distances", max(distance), min(distance))
                    

    mu_lmk_updated[index_initialize] = mu_lmk_initialize

    z = feature[index_update].reshape((-1))
    Nt = index_update.size

    if Nt == 0:
        return pose_updated, mu_lmk_updated, Sigma_updated

    M = feature.shape[0]
    Ht_pose = np.zeros((4*Nt, 6), dtype=np.float64)
    Ht_lmk = np.zeros((4*Nt, 3*M))
    z_tilde = np.zeros((4*Nt), dtype=np.float64)

    for i in range(Nt):
        j = index_update[i]
        lmk_coordinates_cam = cam_T_world @ mu_lmk[j]
        zi_tilde = Ks @ pi(lmk_coordinates_cam)

        dpi = dpi_dq(lmk_coordinates_cam)
        z_tilde[4*i:4*i+4] = zi_tilde
        Hti_pose = -Ks @ dpi @ cam_T_imu @ cdot_hat(inversePose(pose) @ mu_lmk[j])
        Ht_pose[4*i:4*i+4] = Hti_pose

        Hti_lmk = Ks @ dpi @ cam_T_world @ (P.T)
        Ht_lmk[4*i:4*i+4, 3*j:3*j+3] = Hti_lmk

    # print(np.mean((z-z_tilde)**2))

    Ht = np.concatenate((Ht_pose, Ht_lmk), axis=1)
    # v_diag = np.random.normal(loc=0, scale=sigma_v, size=(4))
    # V = np.diag(v_diag ** 2)
    Kt = Sigma @ (Ht.T) @ inv(Ht @ Sigma @ (Ht.T) + np.kron(np.eye(Nt), V))
    pose_updated = pose @ expm(axangle2twist(Kt[:6] @ (z - z_tilde)))


    # Update only useing valid observations
    mu_next = mu_lmk_updated[:, :3] + np.reshape(Kt[6:] @ (z - z_tilde), (-1, 3))
    mu_lmk_updated[index_update, :3] = mu_next[index_update]
    Sigma_next = (np.eye(6+3*M) - (Kt @ Ht)) @ Sigma
    sigma_update_indexes = 3 * index_update + 6
    sigma_update_indexes = np.concatenate((sigma_update_indexes, sigma_update_indexes+1, 
                                           sigma_update_indexes+2, np.arange(6)))
    Sigma_updated[sigma_update_indexes[:, np.newaxis], sigma_update_indexes] \
                = Sigma_next[sigma_update_indexes[:, np.newaxis], sigma_update_indexes]

    return pose_updated, mu_lmk_updated, Sigma_updated