import numpy as np

P = np.eye(3, 4)

sigma_lmk = 0.1
sigma_pose = np.diag([1, 1, 1, 0.05, 0.05, 0.05])
Sigma_lmk0 = np.random.normal(loc=0, scale=sigma_lmk, size=[3])
Sigma_lmk0 = np.diag(Sigma_lmk0)

V = np.diag([6, 6, 6, 6])
# W = np.diag([0.5, 0.5, 0.5, 0.05, 0.05, 0.0001])
W = np.diag([0.1, 0.1, 0.1, 0.001, 0.001, 0.0001])
M = 1200
sigma_v = np.sqrt(100)


# print(V)