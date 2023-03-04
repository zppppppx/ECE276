import time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit, njit
from numba.types import float64, int64
from Config import *
plt.ion()


def tic():
    return time.time()


def toc(tstart, name="Operation"):
    print('%s took: %s sec.\n' % (name, (time.time() - tstart)))


# def mapCorrelation(im, x_im, y_im, vp, xs, ys):
#     '''
#     INPUT 
#     im              the map 
#     x_im,y_im       physical x,y positions of the grid map cells
#     vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
#     xs,ys           physical x,y,positions you want to evaluate "correlation" 

#     OUTPUT 
#     c               sum of the cell values of all the positions hit by range sensor
#     '''
#     nx = im.shape[0]
#     ny = im.shape[1]
#     xmin = x_im[0]
#     xmax = x_im[-1]
#     xresolution = (xmax-xmin)/(nx-1)
#     ymin = y_im[0]
#     ymax = y_im[-1]
#     yresolution = (ymax-ymin)/(ny-1)
#     nxs = xs.size
#     nys = ys.size
#     cpr = np.zeros((nxs, nys))
#     for jy in range(0, nys):
#         y1 = vp[1, :] + ys[jy]  # 1 x 1076
#         iy = np.int16(np.round((y1-ymin)/yresolution))
#         for jx in range(0, nxs):
#             x1 = vp[0, :] + xs[jx]  # 1 x 1076
#             ix = np.int16(np.round((x1-xmin)/xresolution))
#             valid = np.logical_and(np.logical_and((iy >= 0), (iy < ny)),
#                                    np.logical_and((ix >= 0), (ix < nx)))
#             cpr[jx, jy] = np.sum(im[ix[valid], iy[valid]])
#     return cpr

# def mapCorrelation(im, grid_scale, ranges, vp, xs, ys):
#     '''
#     INPUT 
#     im              the map 
#     vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
#     xs,ys           physical x,y,positions you want to evaluate "correlation" 

#     OUTPUT 
#     c               sum of the cell values of all the positions hit by range sensor
#     '''
#     nx = im.shape[0]
#     ny = im.shape[1]
#     xmin = ranges[0, 0] * grid_scale
#     ymin = ranges[1, 0] * grid_scale

#     nxs = xs.size
#     nys = ys.size
#     cpr = np.zeros((nxs, nys))
#     for jy in range(0, nys):
#         y1 = vp[1, :] + ys[jy]  # 1 x 1076
#         iy = np.int16(np.round((y1 - ymin)/grid_scale))
#         for jx in range(0, nxs):
#             x1 = vp[0, :] + xs[jx]  # 1 x 1076
#             ix = np.int16(np.round((x1 - xmin)/grid_scale))
#             valid = np.logical_and(np.logical_and((iy >= 0), (iy < ny)),
#                                    np.logical_and((ix >= 0), (ix < nx)))
#             cpr[jx, jy] = np.sum(im[ix[valid], iy[valid]])
        
#     cpr[np.where(cpr <= 0)] = 0.1
#     return cpr

@jit(numba.float64[::1](numba.float64[::1]), nopython=True)
def rnd(x):
    out = np.empty_like(x)
    np.round(x, 0, out)
    return out

@jit(nopython=True)
def dot(a, b):
    out = np.dot(a, b)
    return out

@numba.njit(numba.float64[::1](numba.float64[:, ::1], numba.float64, numba.int64[:, ::1], numba.float64[:, ::1], numba.float64[::1]))
def mapCorrelation(im, grid_scale, ranges, vp, position):
    '''
    INPUT 
    im              the map 
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
    xs,ys           physical x,y,positions you want to evaluate "correlation" 

    OUTPUT 
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = ranges[0, 0] * grid_scale
    ymin = ranges[1, 0] * grid_scale

    xs = position.copy()[0]
    ys = position.copy()[1]

    ar = np.arange(-grid_mid, grid_mid+1) * grid_scale
    xs = xs + ar
    ys = ys + ar

    theta_ar = np.arange(-theta_mid, theta_mid+1) * theta_delta
    nxs = xs.size
    nys = ys.size
    ntheta = theta_ar.size
    # cpr = np.zeros((nxs, nys))
    cpr = 1
    pos = position.copy().astype(np.float64)
    # rot = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
    angle = 0.
    
    for it in range(ntheta):
        # nrot = rotate_z(theta_ar[it])
        theta = theta_ar[it]
        nrot = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]]).astype(np.float64)
        nvp = nrot.dot(vp)

        for jy in range(0, nys):
            y1 = nvp[1, :] + ys[jy]  # 1 x 1076
            iy = rnd((y1 - ymin)/grid_scale).astype(np.int16)
            for jx in range(0, nxs):
                x1 = nvp[0, :] + xs[jx]  # 1 x 1076
                ix = rnd((x1 - xmin)/grid_scale).astype(np.int16)
                valid = np.logical_and(np.logical_and((iy >= 0), (iy < ny)),
                                    np.logical_and((ix >= 0), (ix < nx)))
                new_cpr = 0
                for iv in range(valid.size):
                    new_cpr += im[ix[iv], iy[iv]]
                
                if new_cpr > cpr:
                    cpr = new_cpr
                    pos = np.array([xs[jx], ys[jy], 0], dtype=np.float64)
                    angle = theta

    return np.array([np.float64(cpr), np.float64(pos[0]), np.float64(pos[1]), np.float64(angle)])


@numba.njit(numba.float64[:, ::1](numba.float64))
def rotate_z(theta):
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]]).astype(np.float64)
    return rot


def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
            (sx, sy)	start point of ray
            (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx  # swap

    if dy == 0:
        q = np.zeros((dx+1, 1))
    else:
        q = np.append(0, np.greater_equal(np.diff(
            np.mod(np.arange(np.floor(dx/2), -dy*dx+np.floor(dx/2)-1, -dy), dx)), 0))
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey+1)
        else:
            y = np.arange(sy, ey-1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex+1)
        else:
            x = np.arange(sx, ex-1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x, y)).astype(np.int32)


def bresenham2D_loop(pos_mobile, lcw):
    fr, ob = np.zeros([2, 1]), np.zeros([2, 1])
    for i in range(lcw.shape[-1]):
        line = bresenham2D(pos_mobile[0], pos_mobile[1], lcw[0, i], lcw[1, i])
        free = line[:, :-1].reshape([2, -1])
        obs = line[:, -1].reshape([2, 1])
        fr = np.concatenate([fr, free], axis=1)
        ob = np.concatenate([ob, obs], axis=1)

    return fr[:, 1:], ob[:, 1:]


def test_bresenham2D():
    import time
    sx = 0
    sy = 1
    print("Testing bresenham2D...")
    r1 = bresenham2D(sx, sy, 10, 5)
    r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]])
    r2 = bresenham2D(sx, sy, 9, 6)
    r2_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                     [1, 2, 2, 3, 3, 4, 4, 5, 5, 6]])
    if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex), np.sum(r2 == r2_ex) == np.size(r2_ex)):
        print("...Test passed.")
    else:
        print("...Test failed.")

    # Timing for 1000 random rays
    num_rep = 1000
    start_time = time.time()
    for i in range(0, num_rep):
        x, y = bresenham2D(sx, sy, 500, 200)
    print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))


def test_mapCorrelation():
    angles = np.arange(-135, 135.25, 0.25)*np.pi/180.0
    ranges = np.load("./code/test_ranges.npy")

    # take valid indices
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # init MAP
    MAP = {}
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -20  # meters
    MAP['ymin'] = -20
    MAP['xmax'] = 20
    MAP['ymax'] = 20
    MAP['sizex'] = int(
        np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']),
                          dtype=np.int8)  # DATA TYPE: char or int8

    # xy position in the sensor frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)

    # convert position in the map frame here
    Y = np.stack((xs0, ys0))

    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res']).astype(np.int16)-1

    # build an arbitrary map
    indGood = np.logical_and(np.logical_and(np.logical_and(
        (xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[indGood[0]], yis[indGood[0]]] = 1

    # x-positions of each pixel of the map
    x_im = np.arange(MAP['xmin'], MAP['xmax']+MAP['res'], MAP['res'])
    # y-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax']+MAP['res'], MAP['res'])

    x_range = np.arange(-0.2, 0.2+0.05, 0.05)
    y_range = np.arange(-0.2, 0.2+0.05, 0.05)

    print("Testing map_correlation with {}x{} cells".format(
        MAP['sizex'], MAP['sizey']))
    ts = tic()
    c = mapCorrelation(MAP['map'], x_im, y_im, Y, x_range, y_range)
    toc(ts, "Map Correlation")

    c_ex = np.array([[3, 4, 8, 162, 270, 132, 18, 1, 0],
                    [25, 1, 8, 201, 307, 109, 5, 1, 3],
                    [314, 198, 91, 263, 366, 73, 5, 6, 6],
                    [130, 267, 360, 660, 606, 87, 17, 15, 9],
                    [17, 28, 95, 618, 668, 370, 271, 136, 30],
                    [9, 10, 64, 404, 229, 90, 205, 308, 323],
                    [5, 16, 101, 360, 152, 5, 1, 24, 102],
                    [7, 30, 131, 309, 105, 8, 4, 4, 2],
                    [16, 55, 138, 274, 75, 11, 6, 6, 3]])

    if np.sum(c == c_ex) == np.size(c_ex):
        print("...Test passed.")
    else:
        print("...Test failed. Close figures to continue tests.")

    # plot original lidar points
    fig1 = plt.figure()
    plt.plot(xs0, ys0, '.k')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Laser reading")
    plt.axis('equal')

    # plot map
    fig2 = plt.figure()
    plt.imshow(MAP['map'], cmap="hot")
    plt.title("Occupancy grid map")

    # plot correlation
    fig3 = plt.figure()
    ax3 = fig3.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(0, 9), np.arange(0, 9))
    ax3.plot_surface(X, Y, c, linewidth=0, cmap=plt.cm.jet,
                     antialiased=False, rstride=1, cstride=1)
    plt.title("Correlation coefficient map")
    plt.show()


def show_lidar():
    angles = np.arange(-135, 135.25, 0.25)*np.pi/180.0
    ranges = np.load("./code/test_ranges.npy")
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, ranges)
    ax.set_rmax(10)
    ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Lidar scan data", va='bottom')
    plt.show()

@numba.jit()
def count_occupancy(lcw, ranges, pos_mobile, free_cells, occupied_cells):
    n_lidar_points = lcw.shape[-1]
    for j in range(n_lidar_points):
        line = bresenham2D(pos_mobile[0], pos_mobile[1], lcw[0, j], lcw[1, j])
        free_cells[line[0, :-1] - ranges[0][0],
                    line[1, :-1] - ranges[1][0]] += 1
        occupied_cells[line[0, -1] - ranges[0]
                        [0], line[1, -1] - ranges[1][0]] += 1
        
    return free_cells, occupied_cells

if __name__ == '__main__':
    show_lidar()
    test_mapCorrelation()
    test_bresenham2D()

    input()
