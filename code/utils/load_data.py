import pickle
import numpy as np


def read_data(fname, key):
    d = []
    with open(fname, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # need for python 3

    data, ts = np.array(d[key], dtype=np.float32), np.array(d['ts'])

    return data, ts


def gen_quaternion(len):
    """
    Generate random unit quaternions with len

    Args:
        len: the data length

    Returns:
        quaternions: generated uniform quaternions
    """
    uniform = lambda x : x/np.sqrt(x.dot(x))
    quaternions = np.random.rand(len, 4)
    # quaternions = uniform(quaternions)
    quaternions = np.array([*map(uniform, quaternions)])
    quaternions[0, :] = np.array([1,0,0,0])

    return quaternions.T

# print(gen_quaternion(3))