import numpy as np

P = np.eye(3, 4)

sigma_lmk = 2
Sigma_lmk0 = np.random.normal(loc=0, scale=sigma_lmk, size=[3])
Sigma_lmk0 = np.diag(Sigma_lmk0)

M = 1000
sigma_v = np.sqrt(100)


# print(V)