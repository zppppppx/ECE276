from utils import *
from model import *
import os
import matplotlib.pyplot as plt
from Config import Config

dataset = "3"
cfile = "./data/cam/cam" + dataset + ".p"
ifile = "./data/imu/imuRaw" + dataset + ".p"
vfile = "./data/vicon/viconRot" + dataset + ".p"

if(os.path.exists(cfile)):
    d_cam, ts_cam = load_data.read_data(cfile, 'cam')
d_imu, ts_imu = load_data.read_data(ifile, 'vals')
d_vic, ts_vic = load_data.read_data(vfile, 'rots')

T = ts_imu.size

print('Loading data has finished!')

# plt.subplot(4, 1, 1)
# plt.plot(ts_imu[0]-ts_imu[0, 0], d_imu[0, :])
# plt.subplot(4, 1, 2)
# plt.plot(ts_imu[0]-ts_imu[0, 0], d_imu[3, :])
# plt.subplot(4, 1, 3)
# plt.plot(ts_imu[0]-ts_imu[0, 0], d_imu[4, :])
# plt.subplot(4, 1, 4)
# plt.plot(d_imu[5, :])
# plt.show()

###############
# Calibration #
###############
euler_x, euler_y, euler_z = calibrate.rot2euler(d_vic)
bias = calibrate.findBias([euler_x, euler_y, euler_z], d_imu, 3)
calibrate.calibrate(d_imu, bias)


################## 
# Initialization #
##################
qT = model.init_qT(d_imu[3:, :], ts_imu)
print(qT.shape)


################
# Optimization #
################
for i in range(Config.epoch_q):
    cost = model.get_cost(qT, d_imu, ts_imu)
    cost_descent = jax.jacrev(model.get_cost)


# plt.plot(qT[1])
# plt.show()

# print(ts_imu[0, 1] - ts_imu[0, 0])

# q = np.array([1,0,0,0])[:, None]
# q_all = np.concatenate([q, quaternions], axis=1)
# print(q_all.shape, q_all)

"""intervals = ts_imu[0, 1:] - ts_imu[0, :-1]
intervals = intervals[None, :]
print(intervals.shape)

cost = model.get_cost(qT, d_imu, ts_imu)
print(cost)
cost_descent = jax.jacrev(model.get_cost)"""

# grad = cost_descent(quaternions, d_imu, ts_imu)
# print(grad.shape)

# hk = model.v_projection(grad, quaternions)
# print(hk[:, :3])

# print(d_imu[0, :] - d_imu[0][0])

# plt.figure(2)
# plt.plot(ts_vic[0] - ts_vic[0][0], det)
# plt.show()



# print(d_imu[:, :3])
# imu = calibrate.calibrate(d_imu, bias=0)
# print(d_imu[:, :3], '\n', imu[:, :3])



# print(ts_imu.shape)
