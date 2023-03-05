import numpy as np

wheel_span = 0.33020
lidar_projected_span = 0.29833
lidar_height = 0.51435
lidar_coor_body = np.array([ lidar_projected_span - wheel_span/2, 0, lidar_height/2]) # the coordinates of the lidar in the car body frame

move_every_tic = 0.254 * np.pi / 360 # the distance of every movement at every tic
grid_scale = 0.02
outer_width = 0.47625 # the width between the outer surface of two parallel wheels
inner_width = 0.31115 # the width between the innner surface of two parallel wheels
L = (outer_width + inner_width) / 2 # the length of axle

freq = 12
ranges = [40, 40]
shift = [int(ranges[0]/grid_scale), int(ranges[1]/grid_scale)]

cpr_grid = 9
grid_mid = int(cpr_grid/2)

theta_range = 5
theta_mid = int(theta_range/2)
theta_delta = 1.5 / 180 * np.pi

N = 100 # the number of the particles
Neff_threshold = 0.2 * N

# the standard deviation of the norm distribution for velocity and angular velocity
sigma_v = 0.04
sigma_w = 0.025

K = np.array([[585.05108211, 0, 242.94140713],
                           [0, 585.05108211, 315.83800193],
                           [0,      0,      1]])
K_inv = np.linalg.inv(K)

oRr = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) # regular to optical
rRo = np.linalg.inv(oRr) # oprical to regular

yaw = 0.021
pitch = -0.36
bRr = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])\
        .dot(np.array([[np.cos(pitch), 0, np.sin(pitch)], [0,1,0], [-np.sin(pitch), 0, np.cos(pitch)]]))