The file structure should be:

|-- root
    |-- code
        |-- train.py: the main process of training the model to do orientation tracking and panorama generation
        |-- Config.py: the configures for training the model, including step_size and epochs
        |-- model
        |   |-- load_data.py: some functions to load data into the main process
            |-- model.py: the most important part including the models and corresponding loss calculation function
            |-- panorama.py: the functions for generating the panorama
        |-- utils
            |-- calibrate.py: some functions for calibration, including find the bias of IMU measurements, transform

            |-- quater.py: basic functions regrading quaternion calculation.

