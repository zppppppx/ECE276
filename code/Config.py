class Config:
    epoch_q = 50 # the epochs to optimize the quaternions
    epoch_phi = 5 # the epochs to optimize the projective angle phi
    epoch = 100

    step_phi = 1e-2 # the step size to optimize the phi
    step_qt = 1e-2