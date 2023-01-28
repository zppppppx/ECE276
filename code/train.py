from utils import *
import os
import matplotlib.pyplot as plt

dataset = "4"
cfile = "./data/cam/cam" + dataset + ".p"
ifile = "./data/imu/imuRaw" + dataset + ".p"
vfile = "./data/vicon/viconRot" + dataset + ".p"

if(os.path.exists(cfile)):
    d_cam, ts_cam = load_data.read_data(cfile, 'cam')
d_imu, ts_imu = load_data.read_data(ifile, 'vals')
d_vic, ts_vic = load_data.read_data(vfile, 'rots')


print('Loading data has finished!')

plt.subplot(4, 1, 1)
plt.plot(ts_imu[0]-ts_imu[0, 0], d_imu[0, :])
plt.subplot(4, 1, 2)
plt.plot(ts_imu[0]-ts_imu[0, 0], d_imu[3, :])
plt.subplot(4, 1, 3)
plt.plot(ts_imu[0]-ts_imu[0, 0], d_imu[4, :])
plt.subplot(4, 1, 4)
plt.plot(ts_imu[0]-ts_imu[0, 0], d_imu[5, :])
plt.show()


# print(d_imu.shape)
T = ts_imu.size

quaternions = load_data.gen_quaternion(T)

print(d_imu[0, :] - d_imu[0][0])

# plt.figure(2)
# plt.plot(ts_vic[0] - ts_vic[0][0], det)
# plt.show()

bias = calibrate.findBias(d_imu, 5)
print(bias)


print(d_imu[:, :10])
