from pr3_utils import *

q = np.random.randn(4)
dq = dpi_dq(q)
# print(q)

# q = pi(q)

# print(q)
@njit
def vstack(a):
    a = tuple(a)
    return np.vstack(a)

# index = np.isnan(a).any(axis=1)
# print(index)

a = np.random.rand(5,2)
print(a)
print(a.reshape((-1)))


a = np.arange(625).reshape((25,25))
index = np.arange(5)
b = a[index[:, np.newaxis], index]
print(b)

