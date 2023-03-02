import numpy as np
from pr2_utils import *
import matplotlib.pyplot as plt
from slam import SLAM
from tqdm import tqdm
from Config import *

for dataset in [20, 21]:
# Initialize the dataset and slam class
    print("########## SLAM on dataset %d with %d particles #########"%(dataset, N))
    Hokuyo_Path = "./data/Hokuyo%d.npz" % dataset
    Encoders_Path = "./data/Encoders%d.npz" % dataset
    IMU_Path = "./data/Imu%d.npz" % dataset

    slam = SLAM(Hokuyo_Path, Encoders_Path, IMU_Path)

    # Initialize the map
    slam.renew_occupancy(slam.particles[0], slam.lidar_coordinates_aligned[:, :, 0])


    w_time_start = 0
    w_time_end = 0
    T_imu = slam.imu_stamps.size

    T_speed = slam.encoder_stamps.size
    positions = np.zeros([3, 1])
    for i in tqdm((range(T_speed-1))):
        # for each time span of adjacent linear velocity, we find corresponding angular velocity series that the 
        # first time span covers the timestamp beginning and the last time span covers the ending
        w_time_start = w_time_end
        while(w_time_end + 1 < T_imu and slam.imu_stamps[w_time_end+1] <= slam.encoder_stamps[i+1]):
            w_time_end += 1

        V_body = slam.V_body[:, i]
        V_time_stamps = slam.encoder_stamps[i:i+2]
        W = slam.imu_angular_velocity[:, w_time_start:w_time_end+1]
        W_time_stamps = slam.imu_stamps[w_time_start:w_time_end+1]


        # For each of the particle
        for j in range(len(slam.particles)):

            # Add motion noise for each of the particle
            V_noise = np.random.normal(loc=0, scale=sigma_v, size=[3])
            V_noise[1:] = 0
            W_noise = np.random.normal(
                loc=0, scale=sigma_w, size=[3, W.shape[-1]])
            W_noise[:-1] = 0
            V_particle = V_body + V_noise
            W_particle = W + W_noise

            slam.particles[j].predict(V_particle, V_time_stamps, W_particle, W_time_stamps)
            
        # Update the weights and find the most_likely_particle
        ind = slam.update(slam.occupancy_map, slam.ranges, slam.lidar_coordinates_aligned[:, :, i])
        # renew the occupancy map and positions using the particle
        slam.renew_occupancy(slam.particles[ind], slam.lidar_coordinates_aligned[:, :, i])
        positions = np.concatenate([positions, slam.particles[ind].position.reshape([3,1])], axis=1)

        # check the weights and resample the particles
        slam.checkAndResample()

    
    plt.figure(figsize=(8, 8))
    plt.imshow(slam.occupancy_map, cmap="hot")
    positions[0, :] -= slam.ranges[0, 0] * grid_scale
    positions[1, :] -= slam.ranges[1, 0] * grid_scale
    positions[0] /= grid_scale
    positions[1] /= grid_scale

    plt.plot(positions[1, :], positions[0, :], '')
    plt.savefig("./figs/slam_d%d_N%d"%(dataset, N), bbox_inches='tight', pad_inches=0.5)
    # plt.show()

    # input()

    np.savez('./results/d%d_N%d.npz'%(dataset, N), 
            oc_map=slam.occupancy_map, map_ranges=slam.ranges, trajectory=positions, grid_scale=grid_scale)
    