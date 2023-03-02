import numpy as np
from utils import *
from pr2_utils import *
from Config import *
import transforms3d as t3d


class Particle:
    """
    This is the class describing the action of particles
    """

    def __init__(self, position: np.array, rot: np.array, weight: np.float32) -> None:
        # initialize the position and rotation matrix of the particle
        self.position = position
        self.rot = rot
        self.qT = t3d.quaternions.mat2quat(rot).reshape(4, 1)
        self.weight = weight

    def predict(self, V_body: np.float32, V_time_stamps: np.array, W, W_time_stamps: np.array):
        """
        Predict the position and the rotation of the particle in the next step

        NOTE: the W_time_stamps is best to have the property that the first two time stamps cover the beginning 
            of the V_time_stamps and the last time stamp and its next time stamp should cover the ending of the 
            V_time_stamps
        Args:
            V_body: the velocity of one recording
            V_time_stamps: the time stamps (only beginning and the ending) of the recording
            W: the angular velocities
            W_time_stamps: corresponding angular velocities' timestamps
        """
        # print("velocity\t", V_body)
        # print("postion\t", self.position)
        # print("time\t", V_time_stamps[1] - V_time_stamps[0])
        V_start_time = V_time_stamps[0]
        V_end_time = V_time_stamps[1]

        W_time_intervals = W_time_stamps[1:] - W_time_stamps[:-1]
        for i in range(W_time_intervals.size):
            qt = motion(self.qT[:, -1], W[:, i], W_time_intervals[i])
            self.qT = np.concatenate([self.qT, qt[:, None]], axis=1)

        self.rot = t3d.quaternions.quat2mat(self.qT[:, -1])
        rots = quat2rot(self.qT)
        self.qT = t3d.quaternions.mat2quat(self.rot).reshape(
            4, 1)  # clear out for efficiency

        V_time = W_time_stamps.copy()
        V_time[0] = V_start_time
        V_time[-1] = V_end_time
        V_time_intervals = V_time[1:] - V_time[:-1]
        V_time_intervals = np.where(V_time_intervals < 0, 0, V_time_intervals)
        for i in range(V_time_intervals.size):
            V = rots[:, :, i].dot(V_body)
            interval = V * V_time_intervals[i]
            # print(interval)
            self.position += interval

    def update(self, occupancy_map: np.array, ranges: np.array, lidar_coordinates: np.array):
        """
        Update the weights of the particle

        Args:
            occupancy_map: the occupancy map
            ranges: the range of the grid-scaled map
            lidar_coordinates: the data obtained from the lidar scan, in the lidar frame
        """
        xs = self.position.reshape([3, -1])[0, :]
        ys = self.position.reshape([3, -1])[1, :]
        lc = lidar_coordinates.copy()
        lc = self.rot.dot(lc) # rotate to standard representation
        cpr = mapCorrelation(occupancy_map, grid_scale, ranges, lc, xs, ys)
        self.weight *= cpr[0,0]

if __name__ == "__main__":
    a = Particle()
    print(a.rot[:, :, 0])
