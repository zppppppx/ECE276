import numpy as np


def findBias(imuData: np.array, threshold: np.float32) -> np.array:
    """
    Use the IMU data to find the bias of the IMU chip, considering the fact that gyroscope data
    sometimes may have a glitch, we need to utilize the accelerometer data to find the last moment
    that the machine stays still.

    Args:
        imuData: collected imudata
        threshold: the threshold to define the first moment of the movement

    Returns:
        bias: the bias found from the imu data
    """
    diff = np.abs(imuData[0, :] - imuData[0, 0])
    indexes = np.argwhere(diff > threshold)

    end = indexes[0][0] - 1
    print(end)
    start = end - 100 if end >= 100 else 0
    
    bias = imuData[:, start:end]
    print(bias.shape)
    bias = np.average(bias, axis=1)

    return bias
    


def calibrate(imuData: np.array, bias: np.array) -> np.array:
    """
    Calibrate the imu data according to fetched bias

    Args:
        imuData: the imu data waiting to be calibrated
        bias: fetched bias

    Return:
        result: calibrated data
    """