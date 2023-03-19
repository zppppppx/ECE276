Running `main.py` would trigger three processes:
    1. SLAM with joint update of pose and landmarks
    2. SLAM with separate update of pose and landmarks
    3. Dead-reckoning with mapping.
Each of the process would produce the process figures, please do not delete the directories

Running `new_main.py` would trigger one process: update the pose and landmarks separately while assuming all the landmarks are independent.

All the configures needed are in constants.py

Main functions are stored in `update.py` and `predict.py`. In `update.py`, the first is updating the landamrks assuming they are independent, the second is updating them assuming they are correlated, the third is updating the pose solely, and the last is updating the pose and the landmarks jointly.