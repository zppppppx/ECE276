import numpy as np
from utils import *
from pr2_utils import *
from Config import *
import transforms3d as t3d
from numba import jit

class Particle:
    """
    This is the class describing the action of particles
    """

    def __init__(self, position: np.array, rot: np.array, weight: np.float64) -> None:
        # initialize the position and rotation matrix of the particle
        self.position = position
        self.rot = rot
        self.qT = t3d.quaternions.mat2quat(rot).reshape(4, 1)
        self.weight = weight

    def predict(self, V_body: np.float64, V_time_stamps: np.array, W, W_time_stamps: np.array):
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
        # xs = self.position.copy().reshape([3, -1])[0, :]
        # ys = self.position.copy().reshape([3, -1])[1, :]

        # ar = np.arange(-grid_mid, grid_mid+1) * grid_scale
        # xs = xs + ar
        # ys = ys + ar


        # lc = lidar_coordinates.copy()
        # lc = self.rot.dot(lc) # rotate to standard representation

        lc = lidar_coordinates.copy()
        # lc = self.rot.dot(lc) # rotate to standard representation
        # cpr = mapCorrelation(occupancy_map, grid_scale, ranges, lc, xs, ys)
        # cpr, npos, nrot = update(self.position, self.rot, occupancy_map, ranges, lc)
        print(self.position.dtype, self.rot.dtype, occupancy_map.dtype, ranges.dtype, lc.dtype)
        print(self.position.shape, self.rot.shape, occupancy_map.shape, ranges.shape, lc.shape)
        res = update(self.position.astype(np.float64), self.rot.astype(np.float64), 
                    occupancy_map.astype(np.float64), ranges.astype(np.float64), lc.astype(np.float64))


        self.weight *= res[0]
        self.position[:2] = res[1:3]
        self.rot = rotate_z(res[-1]).dot(self.rot)


        

@njit(numba.float64[::1](numba.float64[::1], numba.float64[:, ::1], numba.float64[:, ::1], numba.int64[:, ::1], numba.float64[:, ::1]))
def update(position, rot, occupancy_map: np.array, ranges: np.array, lidar_coordinates: np.array):
    lc = lidar_coordinates.copy().astype(np.float64)
    lc = rot.dot(lc) # rotate to standard representation
    # cpr = mapCorrelation(occupancy_map, grid_scale, ranges, lc, xs, ys)
    # print(position.dtype, rot.dtype, occupancy_map.dtype, ranges.dtype, lc.dtype)
    # print(position.shape, rot.shape, occupancy_map.shape, ranges.shape, lc.shape)
    
    res = mapCorrelation(occupancy_map, np.float64(grid_scale), ranges, lc, position)
    # position = npos
    # rot = nrot.dot(rot)

    # return cpr, npos, nrot
    return res

if __name__ == "__main__":
    particle = Particle(np.array([0,0,0]).astype(np.float64), np.diag([1,1,1]).astype(np.float64), 1)
    occ = np.random.rand(30, 30).astype(np.float64)
    ranges = np.array([[-10, 10], [-10, 10]]).astype(np.int64)
    vp = np.random.randn(3, 1000).astype(np.float64)
    position = np.random.randn(3).astype(np.float64)

    particle.update(occ, ranges, vp)
