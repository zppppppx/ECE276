from numba import njit
import numba as nb
import numpy as np
from scipy.linalg import expm
from numba.extending import register_jitable

@njit
def check_all(arr, axis=None):
    """
    This function checks if all elements in a NumPy array are True along a specified axis.
    """
    arr = np.ascontiguousarray(arr) # Ensure that the array is contiguous
    return np.all(arr, axis=axis)

@njit
def check_any(arr, axis=None):
    """
    This function checks if any element in a NumPy array is True along a specified axis.
    """
    arr = np.ascontiguousarray(arr) # Ensure that the array is contiguous
    return np.all(arr, axis=axis)


# @njit((nb.f8[:,:]))
# def nexpm(a):
#     res = expm(a)
#     return res

if __name__ == '__main__':
    # a = np.random.randn(10,8)
    # b = check_all(a>0, axis=1)
    # print(b)
    # a = np.array([[True, False], [True, True], [True, True]])
    # result = check_all(a, axis=1)

    a = np.ones((3,3))
    b = nexpm(a)
    print(b)
    