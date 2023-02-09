from model import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import jax
from utils import *
from scipy import misc
from PIL import Image


# qt = np.array([ 0.4589348, 0.05431292, -0.88377863, 0.07324234], dtype=np.float32)

# qt = np.asarray([[0.30340597, 0.3061103],[0.14721881, 0.14626464],[-0.9299224, -0.9296296],[-0.14668301, -0.14384711]])

# q1 = qt[:, 0]
# q2 = qt[:, 1]


# omega = np.array([0.01407524, 0.18677822, -0.00710806])
# interval = 0.010610103607177734
# f = model.motion(q1, omega, interval)

# def loss(q, f):
#     logterm = 2*quater.log(quater.mul(quater.inv(q), f))
#     loss = jnp.square(quater.norm(logterm)) * 0.5
#     loss = jnp.sum(loss)
#     return loss

# def log_norm(q):
#     return quater.norm(quater.log(q))


# gr = jax.grad(loss)
# grad = gr(q2, f)
# print(grad)


pic = np.random.randint(0, 255, [20, 20, 3], dtype=np.uint8)
print(pic.dtype)
print(pic.shape)
# misc.imsave('img_new_sz.png', pic)
im = Image.fromarray(pic)
im.save('./test.png')