import numpy as np
from pr3_utils import *
from predict import *
from update import *
from tqdm import tqdm

if __name__ == '__main__':

	# Load the measurements
	dataset = '10'
	filename = "./data/%s.npz"%dataset
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	features = features.transpose([2, 1, 0])
	Ks = calc_Ks(K, b)

	# print(Ks, '\n', K)
	obs_times = t.shape[0] - np.sum(np.all(features == [-1, -1, -1, -1], axis=2), axis=0)
	idx = obs_times.argsort()[::-1]
	print(idx.shape)
	
	print("max obs times", obs_times[idx[0]], "min obs times", obs_times[idx[-1]])
	features = features[:, idx[:M], :]
	features = np.asarray(features, order='C')

	gvs = np.r_["0", linear_velocity, angular_velocity].T
	# print(gvs.shape)
	u_hats = axangle2twist(gvs)
	u_curlys = axangle2adtwist(gvs)
	print(u_curlys.shape)
	# print(t.shape, features.shape, linear_velocity.shape, angular_velocity.shape, K.shape, b, imu_T_cam.shape)

	T = t.size
	n_features = features.shape[1]
	mu_lmks = np.full([T, n_features, 4], np.nan)
	Sigma_lmks = np.zeros([T, n_features, 3, 3])
	Sigma_pose = np.eye(6) * sigma_pose
	
	for i in range(n_features):
		Sigma_lmks[0, i] = np.eye(3) * sigma_lmk

	poses = np.zeros([T, 4, 4])
	poses[0] = np.array([[1, 0, 0, 0],
		      			 [0, -1, 0, 0],
						 [0, 0, -1, 0],
						 [0, 0, 0, 1]], dtype=np.float64)
	poses[0] = np.eye(4).astype(np.float64)

	
	pose_res = []

	# This is the approximate way to update the mapping with dead reckoning
	print("Dead reckoning")
	mu_lmks = np.full([n_features, 4], np.nan)
	Sigma_lmks = np.zeros([n_features, 3, 3])
	for i in range(n_features):
		Sigma_lmks[i] = np.eye(3) * sigma_lmk
	poses = np.zeros([T, 4, 4])
	poses[0] = np.eye(4).astype(np.float64)
	for i in tqdm(range(1, T)):
	# for i in range(1, 100):
		u_hat = u_hats[i-1]
		tau = t[0, i]- t[0, i-1] 
		# print(tau)
		poses[i] = predict_pose(poses[i-1], tau, u_hat)

		mu_lmks, Sigma_lmks = update_lmk(mu_lmks, Sigma_lmks, features[i], poses[i], Ks, K, np.float64(b), imu_T_cam)

	print(mu_lmks[:, 3])

	pose_res.append(poses)
	fig, ax = visualize_feature_points_2d(mu_lmks, poses)
	fig.savefig('./figs/%s_deadreckoning'%dataset)

	# This is the approximate way to update the landmarks and pose separately
	print("EKF update separately")
	mu_lmks = np.full([n_features, 4], np.nan)
	Sigma_lmks = np.zeros([n_features, 3, 3])
	for i in range(n_features):
		Sigma_lmks[i] = np.eye(3) * sigma_lmk
	Sigma_pose = np.eye(6) * 0.01
	poses = np.zeros([T, 4, 4])
	poses[0] = np.eye(4).astype(np.float64)
	for i in tqdm(range(1, T)):
	# for i in range(1, 100):
		u_hat = u_hats[i-1]
		u_curly = u_curlys[i-1]
		tau = t[0, i]- t[0, i-1] 
		# print(tau)
		poses[i] = predict_pose(poses[i-1], tau, u_hat)

		Sigma_pose = predict_Sigma(Sigma_pose, tau, u_curly)
		mu_lmks, Sigma_lmks = update_lmk(mu_lmks, Sigma_lmks, features[i], poses[i], Ks, K, np.float64(b), imu_T_cam)
		poses[i], Sigma_pose = update_pose_ekf(poses[i], Sigma_pose, features[i], mu_lmks, Ks, K, b, imu_T_cam)


	pose_res.append(poses)
	fig, ax = visualize_feature_points_2d(mu_lmks, poses)
	fig.savefig('./figs/%s_ekfUpdate'%dataset)
	
	fig, ax = visualize_multiple_trajectories_2d(dead_reckoning=pose_res[0], ekf=pose_res[1])
	fig.savefig('./figs/%s_comarison'%dataset)


