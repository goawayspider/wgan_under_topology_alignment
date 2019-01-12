import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gene_noise import *
#from geom_score import *
#from utils import *


def gen_data(num, name):
    data = []
    if name == 'genus2':
        scale = 2
        centers = [[0.0, 1], [0.0, -1]]
        centers = [(scale * x, scale * y) for x, y in centers]
        for i in range(num):
            point = np.random.randn(2)
            point = point / np.linalg.norm(point)
            center = np.array(random.choice(centers))
            point += center
            data.append(point)
        data = np.array(data)
    elif name == 'normal':
        data = np.random.randn(num, 2)
    elif name == "ring":
        scale = 2.
        center = np.array([0.0, 0.0])
        for i in range(num):
            point = np.random.randn(2)
            point = point / np.linalg.norm(point) + center 
            data.append(point)
        data = np.array(data)
    elif name == "hdisk":
        center = np.array([0.0, 0.0])
        data = np.random.randn(num, 2)
        norms = np.linalg.norm(data, axis = 1)
        # print(np.sum(norms >= 1.0)/float(num));
        data[norms >= 1.0, :] = 0
        data = np.array(data) + center
    elif name == "disk":
        iter = 0
        while(iter < num):
            point = np.random.randn(2)
            if(np.linalg.norm(point) <= 1):
                data.append(point)
                iter += 1
        data = np.array(data)
    elif name == "atomdisk":
        iter = 0
        center = np.array([0.0, 0.0])
        while(iter < num - 1):
            point = np.random.randn(2)
            if(np.linalg.norm(point) <= 1):
                data.append(point)
                iter += 1
        data.append(center)
        data = np.array(data)
    elif name == "3dnormal":
        data = np.random.randn(num , 3)
    elif name == "torus":
        X = lambda phi, theta: (2 + np.cos(phi)) * np.cos(theta)
        Y = lambda phi, theta: (2 + np.cos(phi)) * np.sin(theta)
        Z = lambda phi, theta: np.sin(phi)
        for i in range(num):
            phi = np.random.rand(1)*2*np.pi
            theta = np.random.rand(1)*2*np.pi
            data.append([X(phi, theta), Y(phi, theta), Z(phi, theta)])
        data = np.array(data).squeeze()
    elif name == "sphere":
        for i in range(num):
            point = np.random.randn(3)
            point = point / np.linalg.norm(point)
            data.append(point)
        data = np.array(data)
    elif name == 'hollow':
        iter = 0
        while(iter < num):
            point = np.random.randn(2)
            if(np.linalg.norm(point) <= 1 and np.linalg.norm(point) > 0.3):
                data.append(point)
                iter += 1
        data = np.array(data)
    return data

def main():
    data = gen_data(1000,'ring')
    gene = Gene_noise(data,0.01).get_noise(1000)

    plt.scatter(data[:, 0], data[:, 1], c='orange', marker='+')
    plt.scatter(gene[:, 0], gene[:, 1], c='r', marker='+')
    plt.savefig("result/toy2.png")

if __name__ == '__main__':
    main()





