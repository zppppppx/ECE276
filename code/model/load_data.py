import pickle
import numpy as np
import os.path as path


def read_data(fname, key, dtype=np.float32):
    d = []
    with open(fname, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # need for python 3

    data, ts = np.array(d[key], dtype=dtype), np.array(d['ts'])

    return data, ts


def gen_quaternion(len) -> np.array:
    """
    Generate random unit quaternions with len

    Args:
        len: the data length

    Returns:
        quaternions: generated uniform quaternions
    """
    def uniform(x): return x/np.sqrt(x.dot(x))
    quaternions = np.random.rand(len, 4)
    # quaternions = uniform(quaternions)
    quaternions = np.array([*map(uniform, quaternions)])
    # quaternions[0, :] = np.array([1,0,0,0])

    return quaternions.T

# print(gen_quaternion(3))


def save_subfig(fig, ax, save_path, fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path.join(save_path, fig_name), bbox_inches=extent)
