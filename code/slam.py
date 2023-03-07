import numpy as np
from Config import *
from pr2_utils import *
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from particle import *
from utils import *
from tqdm import tqdm
from scipy.special import logsumexp

class SLAM:
    """
    A class describing the process of SLAM

    Attributes:
        lidar_angle_min: the minimum angle of the lidar
        lidar_angle_max: the maximum angle of the lidar
        lidar_angle_increment: angle increment
        lidar_range_min: the minimum range, and the range that is smaller than this should be discarded
        lidar_range_max: the maximum range, and the range that is bigger than this should be discarded
        lidar_ranges: the collected data
        lidar_stamps: the time stamps of each data collection
        lidar_coordinates: the Cartesian coordinates transformed from the lidar range data
        laser_num: the number of lasers that the lidar hold

        encoder_counts: the collected encoder counts
        encoder_stamps: the time stamps of each collection
        V_r: the velocity of the right wheel
        V_l: the velocity of the left wheel
        V_body: the velocity of the vehicle in the body frame, appending two axes to represent 3D velocity


        imu_angular_velocity: angular velocity read from IMU
        imu_linear_acceleration: linear accerleration read from IMU
        imu_stamps: the time stamps of every data collection


        ranges: the occupancy map range, with the unit of grid scale in Config.py
        occupancy_map: the occupancy map when the optimization goes on
        occupancy_odds: the log odds of cells
    """

    def __init__(self, Hokuyo_Path, Encoder_Path, IMU_Path) -> None:
        """
        Initialize the class, first load the datasets and complete necessary 
        """
        self.__load_lidar(Hokuyo_Path)
        self.__load_encoders(Encoder_Path)
        self.__load_imu(IMU_Path)

        self.ranges = np.array([[0, 0], [0, 0]], dtype=np.int32)
        self.occupancy_map = np.array(
            [[1]], dtype=np.float32)  # 1 is default obstacle
        self.occupancy_odds = np.array([[0]], dtype=np.float32)

        self.align_lidar_and_encoder()

        self.particles = [Particle(np.array([0, 0, 0], dtype=np.float32),
                                   np.diag([1, 1, 1]).astype(np.float32), 1/N) for i in range(N)]

    def __load_lidar(self, Hokuyo_Path):
        """
        Load the lidar data
        """
        with np.load(Hokuyo_Path) as data:
            # start angle of the scan [rad]
            self.lidar_angle_min = data["angle_min"]
            # end angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"]
            # angular distance between measurements [rad]
            self.lidar_angle_increment = data["angle_increment"]
            self.lidar_range_min = data["range_min"]  # minimum range value [m]
            self.lidar_range_max = data["range_max"]  # maximum range value [m]
            # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_ranges = data["ranges"]
            # acquisition times of the lidar scans
            self.lidar_stamps = data["time_stamps"]

        # Drop the invalid items
        self.lidar_ranges = np.where(
            self.lidar_ranges <= self.lidar_range_min, np.nan, self.lidar_ranges)
        self.lidar_ranges = np.where(
            self.lidar_ranges >= self.lidar_range_max, np.nan, self.lidar_ranges)

        self.get_coordinates()

    def __angle_vec(self):
        """
        Generate the angle vector using minimum and maximum angle and the angle increment
        """
        self.laser_num = np.round((self.lidar_angle_max-self.lidar_angle_min)
                                  / self.lidar_angle_increment[0, 0]).astype(np.int32) + 1
        angles = np.array([self.lidar_angle_min + i * self.lidar_angle_increment[0, 0]
                          for i in range(self.laser_num)])

        return angles[None, :]

    def get_coordinates(self):
        """
        Get the cartesian coordinates of all lidar data in the lidar sensor frame.

        Returns:
            coordinates: the Cartesian coordinates transformed from lidar data
                with the shape 3 x #lidar_points x #time_stamps
        """
        angles = self.__angle_vec()
        cosine = np.cos(angles)
        sine = np.sin(angles)
        phi = np.zeros_like(angles)

        multi = np.concatenate([cosine, sine, phi], axis=0)[:, :, None]
        stamp_num = self.lidar_stamps.size
        multi = np.tile(multi, [1, 1, stamp_num])

        self.lidar_ranges = self.lidar_ranges[None, :]
        self.lidar_coordinates = self.lidar_ranges * multi + lidar_coor_body.reshape([3, 1, 1])

    def __load_encoders(self, Encoder_Path):
        """
        Load the encoder data
        """
        with np.load(Encoder_Path) as data:
            self.encoder_counts = data["counts"]  # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"]  # encoder time stamps

        self.get_speed()

    def get_speed(self):
        """
        Use encoder counts and timestamps to calculate the speed of each wheel
        """
        move_right = (
            self.encoder_counts[0, :] + self.encoder_counts[2, :]) / 2 * move_every_tic
        move_left = (
            self.encoder_counts[1, :] + self.encoder_counts[3, :]) / 2 * move_every_tic
        time_invervals = self.encoder_stamps[1:] - self.encoder_stamps[:-1]
        time_invervals = np.concatenate([np.array([0.025]), time_invervals])

        self.V_r = move_right / time_invervals
        self.V_l = move_left / time_invervals
        self.V_body = (self.V_r + self.V_l) / 2
        zeros = np.zeros_like(self.V_body[None])
        zeros = np.tile(zeros, [2, 1])
        self.V_body = np.concatenate([self.V_body[None], zeros], axis=0)

    def speed2world(self, V_body, R):
        """
        Use the orientations to transform the speed vectors to world frame, since the sensor data
        has not been synchronized, we use the closest-in-the-past strategy.
        """
        angular_index = 0
        T_angular = self.imu_stamps.size

        T_speed = self.encoder_stamps.size
        for i in range(T_speed):
            while(angular_index + 1 < T_angular and self.imu_stamps[angular_index+1] <= self.encoder_stamps[i]):
                angular_index += 1

            rot = R[:, :, angular_index]
            V_body[:, i] = rot.dot(V_body[:, i])

        return V_body

    def align_lidar_and_encoder(self):
        """
        Align the lidar data and the encoder data using the closest-in-the-past strategy. This is to 
        set the position as the standard timeline.
        """
        lidar_index = 0
        T_lidar = self.lidar_stamps.size

        T_encoder = self.encoder_stamps.size
        self.lidar_coordinates_aligned = np.zeros(
            [3, self.laser_num, T_encoder])
        for i in range(T_encoder):
            while(lidar_index + 1 < T_lidar and self.lidar_stamps[lidar_index+1] <= self.encoder_stamps[i]):
                lidar_index += 1

            self.lidar_coordinates_aligned[:, :, i] = self.lidar_coordinates[:, :, lidar_index]

    def __load_imu(self, IMU_Path):
        """
        Load the IMU data
        """
        with np.load(IMU_Path) as data:
            # TODO: low pass filter the angular velocity
            # angular velocity in rad/sec
            self.imu_angular_velocity = data["angular_velocity"]
            self.low_pass()
            # self.imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
            # acquisition times of the imu measurements
            self.imu_stamps = data["time_stamps"]

    def low_pass(self):
        """
        Low pass filter the angular velocity with frequency freq (10 Hz by default) TODO
        """
        self.imu_angular_velocity[:2, :] = 0
        # print(self.imu_angular_velocity[-1, :].shape)

        self.imu_angular_velocity[-1, :] = butter_lowpass_filter(self.imu_angular_velocity[-1, :], freq, 100)
        # print(self.imu_angular_velocity[-1, :].shape)

    def get_orientations(self, anguler_velocity):
        """
        Use the angular velocity to calculate the orientation of the mobile
        """
        qT = np.array([1, 0, 0, 0], dtype=np.float32)[:, None]
        T = self.imu_stamps.size - 1
        intervals = self.imu_stamps[1:] - self.imu_stamps[:-1]

        for i in range(T):
            qt = motion(qT[:, -1], anguler_velocity[:,
                        i], intervals[i])[:, None]
            qT = np.concatenate([qT, qt], axis=1)

        R = quat2rot(qT)

        return R

    def get_grid_range(self, coordinates):
        """
        Use the coordinates to calculate the range of the map

        Args:
            coordinates: the 3D or 2D coordinates, and we will only use the x and y scale

        Returns:
            ranges: the range of the coordinates, ranges[0] is the x's range and ranges[1] the y's, one unit stands for one grid
        """
        x = coordinates[0, :]
        y = coordinates[1, :]
        range_x = np.array([np.nanmin(np.nanmin(x)), np.nanmax(np.nanmax(x))])
        range_y = np.array([np.nanmin(np.nanmin(y)), np.nanmax(np.nanmax(y))])

        ranges = np.vstack(
            [np.round(range_x), np.round(range_y)]).astype(np.int32)

        return ranges

    def get_trajectory(self, noised=True):
        """
        Get the trajectory out of the velocity and angular velocity, if noised is true, the velocity and 
        angular velocity would be biased using norm distribution with average 0 and sigma from Config.py

        NOTE: this method directly adds noise to the velocity and angular velocity in one step
        """
        V = self.V_body.copy()
        W = self.imu_angular_velocity.copy()
        if noised:
            v_noise = np.random.normal(
                loc=0, scale=sigma_v, size=[3, V.shape[-1]])
            v_noise[1:, :] = 0  # set the y and z axis of the velocity as 0
            w_noise = np.random.normal(
                loc=0, scale=sigma_w, size=[3, W.shape[-1]])
            # set the x and y axis of the angular velocity as 0
            w_noise[:-1, :] = 0

            V += v_noise
            W += w_noise

        R = self.get_orientations(W)
        V_world = self.speed2world(V, R)

        mobile_pos = np.array([0, 0, 0]).reshape([3, 1])
        intervals = self.encoder_stamps[1:] - self.encoder_stamps[:-1]
        for i in range(intervals.size):
            pos = mobile_pos[:, -1] + intervals[i]*V_world[:, i]
            mobile_pos = np.concatenate([mobile_pos, pos[:, None]], axis=1)

        return mobile_pos/grid_scale

    def predict_only(self):
        """
        Use the particles to do prediction-only
        """
        w_time_start = 0
        w_time_end = 0
        T_imu = self.imu_stamps.size

        T_speed = self.encoder_stamps.size
        trajectories = np.zeros([N, 3, 1])

        for i in tqdm(range(T_speed-1)):
            w_time_start = w_time_end
            while(w_time_end + 1 < T_imu and self.imu_stamps[w_time_end+1] <= self.encoder_stamps[i+1]):
                w_time_end += 1

            V_body = self.V_body[:, i]
            V_time_stamps = self.encoder_stamps[i:i+2]
            W = self.imu_angular_velocity[:, w_time_start:w_time_end+1]
            W_time_stamps = self.imu_stamps[w_time_start:w_time_end+1]

            positions = np.zeros([N, 3, 1])
            for j in range(len(self.particles)):
                V_noise = np.random.normal(loc=0, scale=sigma_v, size=[3])
                V_noise[1:] = 0
                W_noise = np.random.normal(
                    loc=0, scale=sigma_w, size=[3, W.shape[-1]])
                W_noise[:-1] = 0
                V_particle = V_body + V_noise
                W_particle = W + W_noise

                self.particles[j].predict(
                    V_particle, V_time_stamps, W_particle, W_time_stamps)
                positions[j] = self.particles[j].position.reshape([3, 1])

            trajectories = np.concatenate([trajectories, positions], axis=2)

        print("##################### Predict-Only finished ###################")

        return trajectories

    def renew_occupancy(self, particle: Particle, lidar_coordinates: np.array):
        """
        Get the occupancy out of the position of the mobile and the position of the points scanned by the 
        lidar. Considering that we need to continuously renewing the map, we should renew the range of the map
        at the same time. (Because I think it is better and more chanllenging than just assign a big enough space)

        Args:
            particle: the particle object, containing the pose information
            lidar_coordinates: the lidar data collected when the mobile is at the postion pos_mobile, the shape
                should be [3 x #laser x #particles]

        Returns:
            occupancy_map: a numpy array
        """
        pos_mobile = particle.position.copy()
        rot = particle.rot.copy()
        lc = lidar_coordinates.copy()
        # print(~np.isnan(lc).any(axis=0))
        lc = lc[~np.isnan(lc)].reshape([3, -1]) # drop the invalid items
        # lc = lc[:, ~np.isnan(lc).any(axis=0)] 
        # print(lc.shape)
        lcw = rot.dot(lc) + pos_mobile.reshape([3, -1])  # transform to world frame
        # transform to grid-scaled representation
        pos_mobile /= grid_scale
        lcw /= grid_scale

        coordinates = np.concatenate([pos_mobile.reshape([3, -1]), lcw.reshape([3, -1])], axis=1)
        new_ranges = self.get_grid_range(coordinates)
        old_ranges = self.ranges.copy()

        # new range should be the full union in rectangle space of the two rectangles
        self.ranges = np.array([[np.min([old_ranges[0][0], new_ranges[0][0]]), np.max([old_ranges[0][1], new_ranges[0][1]])],
                                [np.min([old_ranges[1][0], new_ranges[1][0]]), np.max([old_ranges[1][1], new_ranges[1][1]])]])
        # print(np.max(np.max(pos_mobile)), np.max(np.max(lcw)), np.min(np.min(pos_mobile)), np.min(np.min(lcw)))
        if self.ranges[0,0] < old_ranges[0,0] or self.ranges[0,1] > old_ranges[0,1] or self.ranges[1,0] < old_ranges[1,0] \
            or self.ranges[1, 1] > old_ranges[1, 1]:
            old_occupancy_odds = self.occupancy_odds.copy()

            x_shift = old_ranges[0][0] - self.ranges[0][0]
            x_range = old_ranges[0][1] - old_ranges[0][0] + 1

            y_shift = old_ranges[1][0] - self.ranges[1][0]
            y_range = old_ranges[1][1] - old_ranges[1][0] + 1

            # create a new map with the new range and shift the old map to appropriate position
            self.occupancy_odds = np.zeros([self.ranges[0][1] - self.ranges[0][0] + 1,
                                            self.ranges[1][1] - self.ranges[1][0] + 1], dtype=np.float32)
            self.occupancy_odds[x_shift:(x_shift+x_range),
                                y_shift:(y_shift+y_range)] = old_occupancy_odds
            self.occupancy_map = np.zeros_like(self.occupancy_odds, dtype=np.float32)

        # TODO: need to implement the way to sample the occupied or free cells
        # these three variables are the number of corresponding cells
        free_cells = np.zeros_like(self.occupancy_odds, dtype=np.int16)
        occupied_cells = np.zeros_like(self.occupancy_odds, dtype=np.int16)

        # loop over all the valid lasers and count the number of the free cells and occupied cells
        n_lidar_points = lcw.shape[-1]
        for j in range(n_lidar_points):
            line = bresenham2D(pos_mobile[0], pos_mobile[1], lcw[0, j], lcw[1, j])
            free_cells[line[0, :-1] - self.ranges[0][0],
                       line[1, :-1] - self.ranges[1][0]] += 1
            occupied_cells[line[0, -1] - self.ranges[0]
                           [0], line[1, -1] - self.ranges[1][0]] += 1


        # self.occupancy_odds += 2*np.log(4)*occupied_cells
        # self.occupancy_odds -= np.log(4)*free_cells
        self.occupancy_odds[np.where(occupied_cells > free_cells)] += np.log(4)
        self.occupancy_odds[np.where(occupied_cells < free_cells)] -= np.log(4)
        self.occupancy_odds[np.where(self.occupancy_odds > lambda_max)] = lambda_max
        self.occupancy_odds[np.where(self.occupancy_odds < lambda_min)] = lambda_min

        self.occupancy_map[np.where(self.occupancy_odds > 0)] = 1
        self.occupancy_map[np.where(self.occupancy_odds < 0)] = 0

    def dead_reckoning(self):
        # just use one particle
        particle = Particle(np.array([0, 0, 0], dtype=np.float32), np.diag([1,1,1]).astype(np.float32), 1)
        w_time_start = 0
        w_time_end = 0
        T_imu = self.imu_stamps.size

        T_speed = self.encoder_stamps.size
        # positions = np.array

        for i in tqdm(range(T_speed-1)):
        # for i in tqdm(range(500)):
            w_time_start = w_time_end
            while(w_time_end + 1 < T_imu and self.imu_stamps[w_time_end+1] <= self.encoder_stamps[i+1]):
                w_time_end += 1

            V_body = self.V_body[:, i]
            V_time_stamps = self.encoder_stamps[i:i+2]
            W = self.imu_angular_velocity[:, w_time_start:w_time_end+1]
            W_time_stamps = self.imu_stamps[w_time_start:w_time_end+1]

            particle.predict(V_body, V_time_stamps, W, W_time_stamps)
            # if i > 395 and i < 400:
            #     print(particle.position, np.nanmax(np.nanmax(self.lidar_coordinates_aligned[:2, :, i])), np.nanmin(np.nanmin(self.lidar_coordinates_aligned[:2, :, i])))
            #     print(V_body, W)
            slam.renew_occupancy(particle, self.lidar_coordinates_aligned[:, :, i])




    def update(self, occupancy_map, ranges, lidar_coordinates):
        """
        Update the weights of the particles

        Args:
            occupancy_map: the occupancy map
            ranges: the range of the grid-scaled map
            lidar_coordinates: the data obtained from the lidar scan, in the lidar frame
        """
        for i in range(N):
            res = update(self.particles[i].position.astype(np.float64), self.particles[i].rot.astype(np.float64),
                        occupancy_map.astype(np.float64), ranges.astype(np.int64), 
                        lidar_coordinates.astype(np.float64))
            # self.particles[i].weight *= res[0]
            self.particles[i].weight *= np.exp(res[0])
            self.particles[i].position[:2] = res[1:3]
            self.particles[i].rot = rotate_z(res[-1]).dot(self.particles[i].rot.astype(np.float64))
        # for i in range(N):
        #     self.particles[i].update(occupancy_map, ranges, lidar_coordinates)

        # weights = np.array([self.particles[i].weight for i in range(N)])
        # weights_sum = np.sum(weights)

        # for i in range(N):
        #     self.particles[i].weight /= weights_sum  # normalization

        weights = np.array([np.log(self.particles[i].weight) for i in range(N)])
        weights = weights - logsumexp(a=weights)
        for i in range(N):
            self.particles[i].weight = np.exp(weights[i])  # normalization
        

        return np.argmax(weights)

    def checkAndResample(self):
        """
        Check the number of effective particles and if it is lower than the threshold, do stratified resampling.
        """
        weights = np.array([self.particles[i].weight for i in range(N)])
        Neff = 1 / np.sum(weights**2)
        # print(Neff)
        if (Neff < Neff_threshold):
            j, c = 0, weights[0]
            for k in range(N):
                u = np.random.rand() * 1 / N
                beta = u + k/N
                while (beta > c):
                    j += 1
                    c += weights[j]
                self.particles[k] = Particle(
                    self.particles[k].position, self.particles[k].rot, 1/N)
        else:
            return


if __name__ == "__main__":
    dataset = 20
    Hokuyo_Path = "./data/Hokuyo%d.npz" % dataset
    Encoders_Path = "./data/Encoders%d.npz" % dataset
    IMU_Path = "./data/Imu%d.npz" % dataset

    slam = SLAM(Hokuyo_Path, Encoders_Path, IMU_Path)

    pos0 = np.array([0, 0, 0]).reshape([3, 1])
    lidar = slam.lidar_coordinates[:, :, 0].reshape([3, -1, 1])
    print(lidar.shape)

    print((slam.imu_stamps[-1]-slam.imu_stamps[0])/slam.imu_stamps.size)

    # slam.get_occupancy(pos0, lidar)

    # plt.imshow(slam.occupancy_map.transpose(1,0))
    # # plt.imshow(np.zeros([8, 8], dtype=np.float32))
    # plt.show()
    # input()

    # print(slam.imu_angular_velocity.shape)
    # print(slam.lidar_coordinates.shape)
    # print(slam.encoder_stamps.shape)
    # print(slam.lidar_coordinates_aligned[:, 400, 4000])

    # plot_trajectory(slam, 5)
    # input()

    # plot_particle_trajectory(slam)

    slam.renew_occupancy(slam.particles[0], slam.lidar_coordinates_aligned[:, :, 0])
    slam.dead_reckoning()
    plt.imshow(slam.occupancy_map)
    plt.show()
    # for i in range(N):
    #     print(slam.particles[i].weight)
    # slam.update(slam.occupancy_map, slam.ranges, slam.lidar_coordinates[:, :, 0])
    # for i in range(N):
    #     print(slam.particles[i].weight)

    input()
