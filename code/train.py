from utils import *
from model import *
import os
import matplotlib.pyplot as plt
from Config import Config
from tqdm import tqdm
from PIL import Image


for i in range(9, 12):
    dataset = str(i)

    cfile = "./data/cam/cam" + dataset + ".p"
    ifile = "./data/imu/imuRaw" + dataset + ".p"
    vfile = "./data/vicon/viconRot" + dataset + ".p"

    if(os.path.exists(cfile)):
        d_cam, ts_cam = load_data.read_data(cfile, 'cam', np.uint8)
    d_imu,  ts_imu = load_data.read_data(ifile, 'vals')
    if(os.path.exists(vfile)):
        d_vic, ts_vic = load_data.read_data(vfile, 'rots')

    T = ts_imu.size

    print(d_cam.shape)

    print('Loading data has finished!')
    intervals = (ts_imu[0, 1:] - ts_imu[0, :-1])[None, :]

    ###############
    # Calibration #
    ###############
    if(os.path.exists(vfile)):
        euler_x, euler_y, euler_z = calibrate.rot2euler(d_vic)
        bias = calibrate.findBias([euler_x, euler_y, euler_z], d_imu, 3)
    else:
        bias = calibrate.estBias(d_imu, (0, 200))
    calibrate.calibrate(d_imu, bias)
    print("Calibration has finished!")


    ################## 
    # Initialization #
    ##################
    qT = model.init_qT(d_imu[3:, :], ts_imu)
    # Add some noise to the generated quaternions for better optimization effect
    ns = load_data.gen_quaternion(T-1) * 0.05 
    qT += ns

    print("qT has been initialized!")


    ################
    # Optimization #
    ################
    # for i in range(1):
    ##################################################################
    # for i in range(Config.epoch_q):
    #     # cost = model.get_cost(qT, d_imu, intervals)
    #     cost_descent = jax.jacrev(model.get_cost)
    #     grad = cost_descent(qT, d_imu, intervals)
    #     grad = grad.at[jnp.isnan(grad)].set(0.01)
    #     # print(grad)
    #     nk = model.v_projection(grad, qT)
    #     # print("original cost: ", cost)

    #     # phi_k = np.random.rand(1, T-1) * 2* np.pi
    #     phi_k = np.zeros([1, T-1])
    #     # print(nk, grad)
    #     for i in range(Config.epoch_phi):
    #     # for j in range(1):
    #         cost = model.get_cost_phi(phi_k, qT, nk, d_imu, intervals)
    #         print("Inside optimization:", cost)
    #         cost_descent = jax.grad(model.get_cost_phi)
    #         phi_grad = cost_descent(phi_k, qT, nk, d_imu, intervals)
    #         phi_grad = phi_grad.at[jnp.isnan(phi_grad)].set(0.01)
    #         phi_k -= Config.step_phi*phi_grad
    #     print("After optimization: ", cost)
    #     qT = qT*np.cos(phi_k) + nk*np.sin(phi_k)
    ###################################################################


    # for i in range(30):
    #     cost = model.l_observ(qT, d_imu[:3, :-1])
    #     cost_descent = jax.jacrev(model.l_observ)
    #     grad = cost_descent(qT, d_imu[:3, :-1])

    #     print("Epoch {}: cost {}".format(i, cost))
    #     qT -= Config.step_qt * grad
    #     norm = np.linalg.norm(qT, axis=0)[None, :]
    #     # print(norm.shape)
    #     qT = qT/norm


    # for i in range(20):
    #     cost = model.get_cost(qT, d_imu, intervals)
    #     cost_descent = jax.jacrev(model.get_cost)
    #     grad = cost_descent(qT, d_imu, intervals)
    #     grad = grad.at[jnp.isnan(grad)].set(0.01)
    #     if(i % 5 == 4):
    #         print("Epoch {}: cost {}".format(i, cost))
    #     qT -= Config.step_qt * grad
    #     norm = np.linalg.norm(qT, axis=0)[None, :]
    #     # print(norm.shape)
    #     qT = qT/norm

    ######################################################################################################################
    # gr_motion = jax.jacrev(model.l_motion)
    # gr_observ = jax.jacrev(model.l_observ)
    # for i in range(150):
    #     # cost_motion = model.l_motion(qT, d_imu[3:, :-1], intervals)
    #     # cost_observ = model.l_observ(qT, d_imu[:3, :-1])
    #     cost_motion = model.l_motion(qT, d_imu[3:, 1:], intervals)
    #     cost_observ = model.l_observ(qT, d_imu[:3, 1:])
    #     cost = cost_motion + cost_observ

    #     grad_mo = gr_motion(qT, d_imu[3:, :-1], intervals)
    #     grad_ob = gr_observ(qT, d_imu[:3, :-1])

    #     grad_mo = grad_mo.at[jnp.isnan(grad_mo)].set(0)

    #     grad = grad_ob + grad_mo
        
    #     if(i % 5 == 4):
    #         print("Epoch {}: total cost: {}, motion loss: {}, observ loss: {}".format(i, cost, cost_motion, cost_observ))
    #     qT -= Config.step_qt * grad
        
    #     norm = np.linalg.norm(qT, axis=0)[None, :]
    #     # print("before", jnp.isnan(norm).any(), jnp.isnan(qT).any())
    #     # print(norm)
    #     qT = qT/norm

    #     # print("after", jnp.isnan(norm).any(), jnp.isnan(qT).any())
        










    print("Optimizing dataset #{}".format(dataset))
    gr_motion = jax.jacrev(model.l_motion)
    gr_observ = jax.jacrev(model.l_observ)
    cost_descent = jax.grad(model.get_cost_phi)
    for i in range(Config.epoch_q):
        cost_motion = model.l_motion(qT, d_imu[3:, :-1], intervals) * 10
        cost_observ = model.l_observ(qT, d_imu[:3, :-1])
        cost = cost_motion + cost_observ

        grad_mo = gr_motion(qT, d_imu[3:, :-1], intervals) * 10
        grad_ob = gr_observ(qT, d_imu[:3, :-1])

        grad_mo = grad_mo.at[jnp.isnan(grad_mo)].set(0)

        grad = grad_ob + grad_mo
        nk = model.v_projection(grad, qT)

        phi_k = np.zeros([1, T-1])
        # print(nk, grad)
        for j in range(Config.epoch_phi):
        # for j in range(1):
            cost = model.get_cost_phi(phi_k, qT, nk, d_imu, intervals)
            print("Inside optimization:", cost)
            
            phi_grad = cost_descent(phi_k, qT, nk, d_imu, intervals)
            phi_grad = phi_grad.at[jnp.isnan(phi_grad)].set(0.1)
            phi_k -= Config.step_phi*phi_grad
        # print("After optimization: ", cost)
        qT = qT*np.cos(phi_k) + nk*np.sin(phi_k)

    for i in range(50):
        # Modified the ratio of the cost to better train the motion model 
        cost_motion = model.l_motion(qT, d_imu[3:, :-1], intervals) * 10
        cost_observ = model.l_observ(qT, d_imu[:3, :-1])
        cost = cost_motion + cost_observ

        grad_mo = gr_motion(qT, d_imu[3:, :-1], intervals) * 10
        grad_ob = gr_observ(qT, d_imu[:3, :-1])
        grad_mo = grad_mo.at[jnp.isnan(grad_mo)].set(0)
        grad = grad_ob + grad_mo
        
        print("Epoch {}: total cost: {}, motion loss: {}, observ loss: {}".format(i, cost, cost_motion, cost_observ))
        qT -= Config.step_qt * grad
        norm = np.linalg.norm(qT, axis=0)[None, :]
        qT = qT/norm



    nx, ny, nz = calibrate.quat2euler(qT)

    fig, axes = plt.subplots(2, 2)
    fig.canvas.set_window_title("Data set: #" + dataset)
    fig.set_size_inches(12, 10, forward=True)

    if(os.path.exists(vfile)):
        axes[0, 0].plot(ts_vic[0], euler_x, 'b', label='vicon')
        axes[0, 1].plot(ts_vic[0], euler_y, 'b', label='vicon')
        axes[1, 0].plot(ts_vic[0], euler_z, 'b', label='vicon')

    axes[0, 0].plot(ts_imu[0], nx, 'r', label='opt', alpha=0.7)
    axes[0, 1].plot(ts_imu[0], ny, 'r', label='opt', alpha=0.7)
    axes[1, 0].plot(ts_imu[0], nz, 'r', label='opt', alpha=0.7)

    

    axes[0, 0].set_title('Roll')
    axes[0, 1].set_title('Pitch')
    axes[1, 0].set_title('Yaw')

    for i in range(3):
        axes[i//2, i%2].legend()
        axes[i//2, i%2].set_xlabel('time / s')
        axes[i//2, i%2].set_ylabel('angle / deg')

    for idx, name in enumerate(['Roll', 'Pitch', 'Yaw']):
        load_data.save_subfig(fig, axes[idx//2, idx%2], './figs', name+'_'+dataset)

    if(os.path.exists(cfile)):
        pic = panorama.genPanorama(d_cam, ts_cam, ts_imu, qT)
        axes[1, 1].imshow(pic)
        axes[1, 1].set_title("Panorama")

        # load_data.save_subfig(fig, axes[1, 1], './figs', 'panorama_' + dataset)

        im = Image.fromarray(pic)
        im.save('./figs/panorama_{}.png'.format(dataset))


    # plt.show()