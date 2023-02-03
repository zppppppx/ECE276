from model import *
import numpy as np
import matplotlib.pyplot as plt


omega = np.array([1, 2, 3])
interval = 2
a = np.concatenate([[0], interval*omega/2])
qt = np.array([1,2,3,4])

motion = model.motion(qt, omega, interval)

print(motion)

plt.subplot(2, 3, 1)
plt.plot(1)
plt.subplot(2, 3, 2)
plt.plot(2)
plt.subplot(2, 3, 3)
plt.plot(3)
plt.show()