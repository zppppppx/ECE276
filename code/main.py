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

	to_body = np.array([[1, 0, 0],
						[0, -1, 0],
						[0, 0, -1]], dtype=np.float64)
	linear_velocity = to_body @ linear_velocity
	angular_velocity = to_body @ angular_velocity
	
	n_all = features.shape[1]
	Ks = calc_Ks(K, b)

	# valid_idx = feature_downsample(features, K, b)
	# print("valid indexes number:", valid_idx.size)
	# features = features[:, valid_idx]

	# print(Ks, '\n', K)
	obs_times = t.shape[0] - np.sum(np.all(features == [-1, -1, -1, -1], axis=2), axis=0)
	idx = obs_times.argsort()[::-1]
	print(idx.shape)
	
	print("max obs times", obs_times[idx[0]], "min obs times", obs_times[idx[-1]])
	interval = int(n_all/M)
	features = features[:, 0::interval, :]
	# features = features[:, idx[:M]]
	features = np.asarray(features, order='C')

	print(features.shape)

	gvs = np.r_["0", linear_velocity, angular_velocity].T
	# print(gvs.shape)
	u_hats = axangle2twist(gvs)
	u_curlys = axangle2adtwist(gvs)
	print(t[0,1]-t[0,0])
	# print(t.shape, features.shape, linear_velocity.shape, angular_velocity.shape, K.shape, b, imu_T_cam.shape)

	T = t.size
	n_features = features.shape[1]
	mu_lmks = np.full([n_features, 4], np.nan)
	Sigma_pose0 = np.eye(6) * sigma_pose * 0.001
	Sigma = np.eye(6 + 3*n_features) * 4
	Sigma[:6, :6] = Sigma_pose0

	poses = np.zeros([T, 4, 4])
	poses[0] = np.array([[1, 0, 0, 0],
		      			 [0, -1, 0, 0],
						 [0, 0, -1, 0],
						 [0, 0, 0, 1]], dtype=np.float64)
	# poses[0] = np.eye(4).astype(np.float64)
	
	poses_res = []
	
	pos, _, _ = dead_reckoning_visualize(t, linear_velocity, angular_velocity, mu0=poses[0])
	poses_res.append(pos)
	poses_res.append(poses)
	

	############################################
	# This is to update them jointly using EKF #
	############################################
	print("Jointly Update")
	mu_lmks = np.full([n_features, 4], np.nan)
	for i in tqdm(range(1, T)):
	# for i in range(1, T):
		u_hat = u_hats[i-1]
		u_curly = u_curlys[i-1]
		# print(u_hat)
		tau = t[0, i]- t[0, i-1] 
		# print(tau)
		poses[i] = predict_pose(poses[i-1], tau, u_hat)
		Sigma = predict_Sigma_all(Sigma, tau, u_curly, W)
		poses[i], mu_lmks, Sigma = update_slam_ekf(mu_lmks, Sigma, features[i], poses[i], Ks, K, b, imu_T_cam)

		if i % 40 == 39:
			fig, ax = visualize_feature_points_2d(mu_lmks, poses[:i], show=False)
			fig.savefig('./process_slam_joint%s/%s_slam_joint_ekf_%d'%(dataset, dataset, i))
			plt.close(fig)
		
	poses_res[1] = poses
	fig, ax = visualize_feature_points_2d(mu_lmks, poses, show=False)
	fig.savefig('./figs/%s_ekf_slam_joint'%dataset)
	plt.close(fig)

	fig, ax = visualize_multiple_trajectories_2d(show=False, dead_reckoning=poses_res[0], slam_ekf=poses_res[1])
	fig.savefig('./figs/%s_ekf_comparison_joint'%dataset)
	plt.close(fig)


	####################################################
	# This is to update them separately both using EKF #
	####################################################
	print('Separately Update')
	mu_lmks = np.full([n_features, 4], np.nan)
	Sigma_lmks = np.eye(n_features*3) * 5
	Sigma_pose = np.eye(6) * 0.00001
	# for i in range(1, T):
	for i in tqdm(range(1, T)):
		u_hat = u_hats[i-1]
		u_curly = u_curlys[i-1]
		# print(u_hat)
		tau = t[0, i]- t[0, i-1] 
		# print(tau)
		poses[i] = predict_pose(poses[i-1], tau, u_hat)
		Sigma_pose = predict_Sigma(Sigma_pose, tau, u_curly, W)
		poses[i], Sigma_pose = update_pose_ekf(poses[i], Sigma_pose, features[i], mu_lmks, Ks, K, b, imu_T_cam)
		mu_lmks, Sigma_lmks = update_lmk_ekf(mu_lmks, Sigma_lmks, features[i], poses[i], Ks, K, np.float64(b), imu_T_cam)
		
		if i % 40 == 39:
			fig, ax = visualize_feature_points_2d(mu_lmks, poses[:i], show=False)
			fig.savefig('./process_slam_separate%s/%s_slam_separate_ekf_%d'%(dataset, dataset, i))
			plt.close(fig)

	poses_res[1] = poses
	fig, ax = visualize_feature_points_2d(mu_lmks, poses, show=False)
	fig.savefig('./figs/%s_ekf_slam_separate'%dataset)
	plt.close(fig)

	fig, ax = visualize_multiple_trajectories_2d(show=False, dead_reckoning=poses_res[0], slam_ekf=poses_res[1])
	fig.savefig('./figs/%s_ekf_comparison_separate'%dataset)
	plt.close(fig)


	##############################################
	# This is to update only landmarks using EKF #
	##############################################
	print('Update Landmarks only')
	mu_lmks = np.full([n_features, 4], np.nan)
	Sigma_lmks = np.eye(n_features*3) * 5
	Sigma_pose = np.eye(6) * 0.00001
	# for i in range(1, T):
	for i in tqdm(range(1, T)):
		u_hat = u_hats[i-1]
		u_curly = u_curlys[i-1]
		# print(u_hat)
		tau = t[0, i]- t[0, i-1] 
		# print(tau)
		poses[i] = predict_pose(poses[i-1], tau, u_hat)
		mu_lmks, Sigma_lmks = update_lmk_ekf(mu_lmks, Sigma_lmks, features[i], poses[i], Ks, K, np.float64(b), imu_T_cam)
		
		if i % 40 == 39:
			fig, ax = visualize_feature_points_2d(mu_lmks, poses[:i], show=False)
			fig.savefig('./process_mapping%s/%s_mapping_ekf_%d'%(dataset, dataset, i))
			plt.close(fig)

	fig, ax = visualize_feature_points_2d(mu_lmks, poses, show=False)
	fig.savefig('./figs/%s_lmk_update'%dataset)
	plt.close(fig)



	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


