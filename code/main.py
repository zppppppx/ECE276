import numpy as np
from pr3_utils import *
from predict import *
from update import *
from tqdm import tqdm

if __name__ == '__main__':

	# Load the measurements
	filename = "./data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	features = features.transpose([2, 1, 0])
	Ks = calc_Ks(K, b)

	# print(Ks, '\n', K)
	obs_times = t.shape[0] - np.sum(np.all(features == [-1, -1, -1, -1], axis=2), axis=0)
	idx = obs_times.argsort()[::-1]
	print(idx.shape)
	
	print("max obs times", obs_times[idx[0]], "min obs times", obs_times[idx[2000]])
	features = features[:, idx[:M], :]
	features = np.asarray(features, order='C')

	gvs = np.r_["0", linear_velocity, angular_velocity].T
	# print(gvs.shape)
	u_hats = axangle2twist(gvs)
	# print(t.shape, features.shape, linear_velocity.shape, angular_velocity.shape, K.shape, b, imu_T_cam.shape)

	T = t.size
	n_features = features.shape[1]
	mu_lmks = np.full([T, n_features, 4], np.nan)
	# Sigma_lmks = np.zeros([T, n_features, 3, 3])
	Sigma_lmks = np.eye(3*n_features) * 0.01
	
	# for i in range(n_features):
	# 	Sigma_lmks[0, i] = np.eye(3) * 0.01

	poses = np.zeros([T, 4, 4])
	poses[0] = np.array([[1, 0, 0, 0],
		      			 [0, -1, 0, 0],
						 [0, 0, -1, 0],
						 [0, 0, 0, 1]], dtype=np.float64)

	
	
	# for i in tqdm(range(1, T)):
	# # for i in range(1, 100):
	# 	u_hat = u_hats[i-1]
	# 	# print(u_hat)
	# 	tau = t[0, i]- t[0, i-1] 
	# 	# print(tau)
	# 	poses[i] = predict_pose(poses[i-1], tau, u_hat)
	# 	# print(poses[i])
	# 	# print(mu_lmks[i-1].shape, Sigma_lmks[i-1].shape, features[i].shape, poses[i].shape, Ks.shape, K.shape, b.shape, imu_T_cam.shape)
	# 	# print(mu_lmks[i-1].dtype, Sigma_lmks[i-1].dtype, features[i].dtype, poses[i].dtype, Ks.dtype, K.dtype, b.dtype, imu_T_cam.dtype)
	# 	# print(mu_lmks[i, 0])

	# 	mu_lmks[i], Sigma_lmks[i] = update_lmk(mu_lmks[i-1], Sigma_lmks[i-1], features[i], poses[i], Ks, K, np.float64(b), imu_T_cam)
	# 	# print(mu_lmks[i, 0])
		

	# # dead_reckoning_visualize(t, linear_velocity, angular_velocity)
	# # visualize_trajectory_2d(poses)

	# visualize_feature_points_2d(mu_lmks[-1], poses)
	# print(poses[[0, 1000, 2000]])









	# for i in range(1, T):
	for i in tqdm(range(1, T)):
		u_hat = u_hats[i-1]
		# print(u_hat)
		tau = t[0, i]- t[0, i-1] 
		# print(tau)
		poses[i] = predict_pose(poses[i-1], tau, u_hat)
		mu_lmks[i], Sigma_lmks = update_lmk_ekf(mu_lmks[i-1], Sigma_lmks, features[i], poses[i], Ks, K, np.float64(b), imu_T_cam)


	visualize_feature_points_2d(mu_lmks[-1], poses)





	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


