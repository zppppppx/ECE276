from pr3_utils import *

q = np.random.randn(4)
dq = dpi_dq(q)
# print(q)

# q = pi(q)

# poses = np.array([[1, 0, 0, 0],
# 		      			 [0, -1, 0, 0],
# 						 [0, 0, -1, 0],
# 						 [0, 0, 0, 1]], dtype=np.float64)

# print(inversePose(poses))

a = np.arange(25)
print(a.reshape((5,5), order='F'))
print(a.reshape((5,5)))