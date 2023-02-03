class Config:
    epoch_q = 20 # the epochs to optimize the quaternions
    epoch_phi = 10 # the epochs to optimize the projective angle phi
    epoch = 100

    step_phi = 1e-1 # the step size to optimize the phi
    step_qt = 2e-2