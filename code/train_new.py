from utils import *
from model import *
import os
import matplotlib.pyplot as plt
from Config import Config
from tqdm import tqdm

dataset = "1"
cfile = "./data/cam/cam" + dataset + ".p"
ifile = "./data/imu/imuRaw" + dataset + ".p"
vfile = "./data/vicon/viconRot" + dataset + ".p"

if(os.path.exists(cfile)):
    d_cam, ts_cam = load_data.read_data(cfile, 'cam')
d_imu,  ts_imu = load_data.read_data(ifile, 'vals')
d_vic, ts_vic = load_data.read_data(vfile, 'rots')

T = ts_imu.size

print('Loading data has finished!')
intervals = (ts_imu[0, 1:] - ts_imu[0, :-1])[None, :]
# print(intervals)
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
print("Calibration has finished!")
# plt.plot(d_imu[0,:])
# plt.show()

################## 
# Initialization #
##################
# qT = model.init_qT(d_imu[3:, :], ts_imu)
qT = load_data.gen_quaternion(T-1)
q_all = np.concatenate([np.array([1,0,0,0])[:, None], qT], axis=1)
# print(qT.shape)
print("qT has been initialized!")

# nx, ny, nz = calibrate.quat2euler(qT)
# plt.subplot(2, 3, 1)
# plt.plot(euler_x)
# plt.subplot(2, 3, 4)
# plt.plot(nx)
# plt.subplot(2, 3, 2)
# plt.plot(euler_y)
# plt.subplot(2, 3, 5)
# plt.plot(ny)
# plt.subplot(2, 3, 3)
# plt.plot(euler_z)
# plt.subplot(2, 3, 6)
# plt.plot(nz)
# plt.show()


################
# Optimization #
################
# for i in range(1):
##################################################################
# for i in range(Config.epoch_q):
#     # cost = model.get_cost(qT, d_imu, intervals)
#     cost_descent = jax.jacrev(model.get_cost)
#     grad = cost_descent(qT, d_imu, intervals)
#     grad = grad.at[jnp.isnan(grad)].set(0.1)
#     # print(grad)
#     nk = model.v_projection(grad, qT)
#     # print("original cost: ", cost)

#     # phi_k = np.random.rand(1, T-1) * 2* np.pi
#     phi_k = np.zeros([1, T-1])
#     # print(nk, grad)
#     for i in range(Config.epoch_phi):
#     # for j in range(1):
#         cost = model.get_cost_phi(phi_k, qT, nk, d_imu, intervals)
#         print("Inside optimization:", cost)
#         cost_descent = jax.grad(model.get_cost_phi)
#         phi_grad = cost_descent(phi_k, qT, nk, d_imu, intervals)
#         phi_k -= Config.step_phi*phi_grad
#     print("After optimization: ", cost)
#     qT = qT*np.cos(phi_k) + nk*np.sin(phi_k)
###################################################################

for i in range(100):

    cost = model.get_cost_a(q_all, d_imu, intervals)
    cost_descent = jax.jacrev(model.get_cost_a)
    grad = cost_descent(q_all, d_imu, intervals)
    grad = grad.at[jnp.isnan(grad)].set(0.2)
    # print(q_all)
    if(i % 5 == 4):
        print("Epoch {}: cost {}".format(i, cost))
    q_all -= Config.step_qt * grad
    # print(q_all)
    norm = np.linalg.norm(q_all, axis=0)[None, :]
    # print(norm.shape)
    q_all = q_all/norm
    q_all = q_all[:, 1:]
    q_all = np.concatenate([np.array([1,0,0,0])[:, None], q_all], axis=1)
    # q_all = q_all[:].at[0].set(np.array([1,0,0,0]))
nx, ny, nz = calibrate.quat2euler(q_all)


plt.subplot(2, 3, 1)
plt.plot(euler_x)
plt.subplot(2, 3, 4)
plt.plot(nx)
plt.subplot(2, 3, 2)
plt.plot(euler_y)
plt.subplot(2, 3, 5)
plt.plot(ny)
plt.subplot(2, 3, 3)
plt.plot(euler_z)
plt.subplot(2, 3, 6)
plt.plot(nz)
plt.show()

