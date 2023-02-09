The file structure should be:

    |-- root
        |-- code
            |-- train.py
            |-- Config.py
            |-- model
                |-- load_data.py
                |-- model.py
                |-- panorama.py
            |-- utils
                |-- calibrate.py
                |-- quater.py
        |-- data
            |-- cam
            |-- imu
            |-- vicon

train.py: the main process of training the model to do orientation tracking and panorama generation.

Config.py: the configures for training the model, including step_size and epochs.

load_data.py: some functions to load data into the main process.

model.py: the most important part including the models and corresponding loss calculation function.

panorama.py: the functions for generating the panorama.

calibrate.py: some functions for calibration, including finding the bias of IMU measurements, transforming between rotation representations.

quater.py: basic functions regrading quaternion calculation.

> In the train.py, a for-loop is used to train the model for all the datasets by following:
> 1. loading data
> 2. calibration
> 3. training model
> 4. generate panorama

> The execution path should be `code/../`, which is the parent directory of `code`