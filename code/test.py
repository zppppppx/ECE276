from model import *
import numpy as np

omega = np.array([1, 2, 3])
interval = 2
a = np.concatenate([[0], interval*omega/2])
qt = np.array([1,2,3,4])

motion = model.motion(qt, omega, interval)

print(motion)