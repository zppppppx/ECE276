import numpy as np

wheel_span = 0.33020
lidar_projected_span = 0.29833
lidar_height = 0.51435
lidar_coor_body = np.array([ lidar_projected_span - wheel_span/2, 0, lidar_height/2]) # the coordinates of the lidar in the car body frame

move_every_tic = 0.254 * np.pi / 360 # the distance of every movement at every tic
grid_scale = 0.1 # the length of one side of the grid, with the unit meter

outer_width = 0.47625 # the width between the outer surface of two parallel wheels
inner_width = 0.31115 # the width between the innner surface of two parallel wheels
L = (outer_width + inner_width) / 2 # the length of axle

N = 10 # the number of the particles
Neff_threshold = 0.2 * N

# the standard deviation of the norm distribution for velocity and angular velocity
sigma_v = 0.1
sigma_w = 0.02