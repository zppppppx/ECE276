from pr3_utils import *
from scipy.linalg import expm

def predict_pose(mu: np.ndarray, tau: float, u_hat: np.ndarray) -> np.ndarray:
    """
    Predict one step of the robot's pose from previous pose mu.

    Args:
        mu: the pose at time t.
        tau: the time span between two states.
        u_hat: the twist form of the velocity, with shape [4, 4]

    Returns:
        mu_next: the predicted pose at time t+1
    """
    mu_next = mu @ expm(tau*u_hat)

    return mu_next

def predict_Sigma(Sigma: np.ndarray, tau: float, u_curly: np.ndarray, noise=None) -> np.ndarray:
    """
    Predict one step of the covariance matrix Sigma

    Args:
        Sigma: the covariance matrix of the previous state
        tau: the time span between two states.
        u_curly: curly-hat form velocity, with shape [6, 6]
        noise: the covariance matrix of the noise on pose, with shape [6, 6]

    Returns:
        Sigma: the covariance matrix of the next state
    """
    if noise is None:
        noise = np.zeros([6, 6])
    Sigma = expm(-tau*u_curly) @ Sigma @ expm(-tau*u_curly).T + noise

    return Sigma


def predict_Sigma_all(Sigma: np.ndarray, tau: float,u_curly: np.ndarray, noise=None) -> np.ndarray:
    """
    Renew the whole Sigma Sigma = [[Sigma_pose, Cov_pose_lmk], [Cov_lmk_pose, Sigma_lmk]]
    """
    F = expm(-tau*u_curly)
    Sigma_predicted = Sigma.copy()
    if noise is None:
        noise = np.zeros([6, 6])

    Sigma_predicted[:6, :6] = F @ Sigma[:6, :6] @ expm(-tau*u_curly).T + noise
    Sigma_predicted[:6, 6:] = F @ Sigma[:6, 6:]
    Sigma_predicted[6:, :6] = Sigma[6:, :6] @ F.T

    return Sigma_predicted













def dead_reckoning_visualize(time_stamps: np.ndarray, linear_velocity: np.ndarray, angular_velocity: np.ndarray, 
                   mu0: np.ndarray=np.eye(4), Sigma0: np.ndarray=np.eye(6)*0.001):
    gv = np.concatenate([linear_velocity, angular_velocity], axis=0).T # general velocity N*6

    u_hats = axangle2twist(gv)
    # u_curlys = axangle2adtwist(gv)

    time_intervals = time_stamps[0, 1:] - time_stamps[0, :-1]
    poses = mu0[..., None]

    for i in range(time_intervals.size):
        tau = time_intervals[i]
        pose = predict_pose(poses[..., -1], tau, u_hats[i])
        poses = np.concatenate([poses, pose[..., None]], axis=2)

    visualize_trajectory_2d(poses)