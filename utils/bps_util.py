import sys
import numpy as np
import tensorflow as tf
from time import time
import math
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def bps_kdt_dis_encode(bps, point_cloud):
    tree = KDTree(point_cloud)
    encode = [0] * bps.shape[0]
    for bidx, bp in enumerate(bps):
        dis, idx = tree.query([bp], k=1)
        encode[bidx] = dis
        # encode[bidx] = dis**2
    return np.expand_dims(np.array(encode).squeeze(), axis=-1)


def bps_kdt_sub_encode(bps, point_cloud):
    tree = KDTree(point_cloud)
    encode = [0] * bps.shape[0]
    for bidx, bp in enumerate(bps):
        dis, idx = tree.query([bp], k=1)
        encode[bidx] = point_cloud[idx] - bp
        # encode[bidx] = dis**2
    return np.array(encode).squeeze()


def bps_encoded(bps, point_cloud):
    encode = [0] * bps.shape[0]
    for bidx, bp in enumerate(bps):
        min_idx = -1
        min_dis = sys.float_info.max
        for idx, cp in enumerate(point_cloud):
            dis = euclidean_distance(bp, cp)
            if dis < min_dis:
                min_dis = dis
                min_idx = idx
        encode[bidx] = point_cloud[min_idx] - bp
        # encode[bidx] = min_dis
    return np.array(encode).squeeze()


def euclidean_distance(bp, cp):
    return np.sum((bp - cp) ** 2)


# basic point set
def bps_rect_grid(k, r=1):
    curoot = round(k ** (1. / 3))
    linear = np.linspace(-1, 1, curoot)
    xyz = []
    for x in range(curoot):
        for y in range(curoot):
            for z in range(curoot):
                xyz.append((linear[x], linear[y], linear[z]))
    return np.array(xyz)


def bps_ball_grid(k, r=1):
    cube_k = k * (6 / np.pi)  # vol of cube / vol of sphere
    curoot = math.ceil(cube_k ** (1. / 3))
    curoot_xyz = [curoot] * 3
    xyz = []
    count = 0
    while len(xyz) < k:
        xyz = []
        linear_x = np.linspace(-1, 1, curoot_xyz[0])
        linear_y = np.linspace(-1, 1, curoot_xyz[1])
        linear_z = np.linspace(-1, 1, curoot_xyz[2])

        for x in range(curoot):
            for y in range(curoot):
                for z in range(curoot):
                    if linear_x[x] ** 2 + linear_y[y] ** 2 + linear_z[z] ** 2 > r ** 2:
                        continue
                    xyz.append((linear_x[x], linear_y[y], linear_z[z]))

        curoot_xyz[count % 3] += 1
        count += 1
    return np.array(xyz[:k])


def bps_random_uniform_ball(k, r=1):
    radius = np.random.uniform(0.0, r ** (1/3), (k, 1))
    theta = np.random.uniform(0., 1., (k, 1)) * 2 * np.pi
    phi = np.arccos(1 - 2 * np.random.uniform(0.0, 1., (k, 1)))
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.concatenate((x, y, z), axis=-1)

#todo
def bps_hcp(k, r=1):
    pass


# if __name__ == "__main__":
#     xyz = bps_ball_grid(8**3)
#     print(xyz.shape)
#     x, y, z = np.split(xyz, 3, axis=1)
#
#     ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#     #  将数据点分成三部分画，在颜色上有区分度
#     ax.scatter(x, y, z, c='y')  # 绘制数据点
#
#     ax.set_zlabel('Z')  # 坐标轴
#     ax.set_ylabel('Y')
#     ax.set_xlabel('X')
#     plt.show()
#

