import pickle
import sys
import time 
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

dataset="9"
cfile = "./data/cam/cam" + dataset + ".p"
ifile = "./data/imu/imuRaw" + dataset + ".p"
vfile = "./data/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import: "+ dataset)


camData = np.array(camd['cam'])
print('Cam data', camData.shape)

imuData = np.array(imud['vals'])
print('IMU data \n', imuData[:, :18], imuData.shape)

vicData = np.array(vicd['rots'])
tsTime = np.array(vicd['ts'])
tsTime = (tsTime[0, :] - tsTime[0, 0])
print('VIC data ', vicData[:, :, 0])
# print(vicd.keys())

# plt.plot(imuData[0, :])
# plt.show()