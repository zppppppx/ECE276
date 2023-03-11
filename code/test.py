from pr3_utils import *

q = np.random.randn(4)
dq = dpi_dq(q)
# print(q)

# q = pi(q)

# print(q)

a = np.random.randn(5,3)
a = np.zeros([5,3])
# a[0] = np.nan
# print(a[:100])
print((a == 0).all(axis=1))

# index = np.isnan(a).any(axis=1)
# print(index)