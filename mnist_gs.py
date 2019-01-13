import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import gzip
import cPickle as pickle
from geom_score import *
from utils import *
from sklearn import manifold
from gene_noise import *
import threading
import datetime
import Isomap
import cifar_gs as cifar

## ring: 1000 data samples, 10000 noises, d = 2, epsilon = sqrt(1/0.8)
## torus: 6000, 12000, d = 3, epsilon = sqrt(1/0.8)
## mnist: 10000, 20000, d = 20, epsilon = ...
## cifar: 10000, 20000, d = 20, epsilon = 

store = False

data_name = 'torus'
store_path = data_name+"_{}_{}.npy"


img_num = 10000
img_dim = {
    'mnist' : 784,
    'cifar' : 3072,
    'ring': 2,
    'torus': 3
}

noise_dim = {
    'mnist':20,
    'cifar':20,
    'ring': 2,
    'torus': 3
}

def gene_toy(num, name):
    data = []
    if name == "ring":
        scale = 2.
        center = np.array([0.0, 0.0])
        for i in range(num):
            point = np.random.randn(2)
            point = point / np.linalg.norm(point) + center 
            data.append(point)
        data = np.array(data)
    elif name == "torus":
        a = 3
        b = 2
        X = lambda phi, theta: (a + b*np.cos(phi)) * np.cos(theta)
        Y = lambda phi, theta: (a + b*np.cos(phi)) * np.sin(theta)
        Z = lambda phi, theta: b*np.sin(phi)
        for i in range(num):
            phi = np.random.rand(1)*2*np.pi
            theta = np.random.rand(1)*2*np.pi
            data.append([X(phi, theta), Y(phi, theta), Z(phi, theta)])
        data = np.array(data).squeeze()
    return data
        
            


def reduce_dim(data,noise_dim):
    images,_ = data
    
    np.random.shuffle(images)
    print(images.shape)
    # images = images[:img_num]
    # print(images.shape)
    #emb image_num * noise_dim
    # emb = manifold.Isomap(n_neighbors=5, n_components=noise_dim, n_jobs = 10).fit_transform(images)
    #emb,err = manifold.locally_linear_embedding(images,n_neighbors=30,n_components=noise_dim)
    emb = Isomap.Isomap(images, None, img_dim[data_name], noise_dim, 5, eps = 100000)
    #print("Done. Reconstruction error: %g" % err)
    print('done')
    emb = emb.astype('float32')

    return images,emb

_noise_num = 12000
_kappa = 0.3 # to control the threshold to construct the cech complex
tp = datetime.datetime.now().isoformat()
rep = 3000




if data_name == 'mnist':
    with gzip.open('save/mnist.pkl.gz', 'rb') as f:
        train_data, _, _ = pickle.load(f)
elif data_name == 'cifar':
    train_data = cifar.load_batch('save/cifar-10-batches-py/')
elif data_name == 'ring':
    train_data = gene_toy(500, data_name)
elif data_name == 'torus':
    train_data = gene_toy(6000, data_name)



if(data_name in ['mnist', 'cifar']):    
    if not store:
        image,emb = reduce_dim(train_data,noise_dim[data_name])
        np.save(store_path.format("image", img_num), image)
        np.save(store_path.format("emb", img_num), emb)
    else:
        image = np.load(store_path.format("image", img_num))
        emb = np.load(store_path.format("emb", img_num))
else:
    image = train_data
    emb = train_data

print(emb.shape)
print(image.shape)



if(data_name not in ['mnist', 'cifar']):
    _imax = 3
    _alpha = 1.0/32
else:
    _imax = 20
    _alpha = None



    
noise = Gene_noise(emb,alpha = _alpha, kappa = _kappa).get_noise(noise_num = _noise_num)
gaussi_noise = np.random.normal(0,1,(_noise_num, noise_dim[data_name]))


if(data_name == "torus" and False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2])
    # plt.scatter(emb[:, 0], emb[:, 1], c='orange', marker='+')
    # plt.scatter(noise[:, 0], noise[:, 1], c='r', marker='+')
    # plt.savefig("save/toy2.png")
    plt.show()


def store_rlts(data, arr, index, n_job, _gamma):
    out = rlts_pooled(data, i_max = _imax, L_0 = 64, n = rep, n_jobs = n_job, gamma = _gamma)
    arr[index] = out
    


data_parts = [image, emb, noise, gaussi_noise]

# data_parts = data_parts[:-2]
names = ["raw", "mfd", "noise", "gaussian"]
TOTAL = 20
n_jobs= [0.2, 0.2, 0.4, 0.4]
rlts_arr = [None] * len(data_parts)

threads = [None]* len(data_parts)

for i in range(len(data_parts)):
    t = threading.Thread(target = store_rlts, name = names[i], args = (data_parts[i], rlts_arr, i, int(n_jobs[i]*TOTAL), _alpha))
    threads[i] = t
    t.start()


# wait
for i in range(len(data_parts)):
    threads[i].join()


geom_scores = []
for i in range(1, len(data_parts)):
    geom_scores.append(geom_score(rlts_arr[0], rlts_arr[i]))
    print("{}=={} = {}".format(names[0], names[i] ,geom_scores[-1]))


for i in range(len(data_parts)):
    np.save('save/'+store_path.format(names[i], "rlts"), rlts_arr[i])
    
colors = ['b', 'r', 'y', 'c']

for i in range(len(data_parts)):
    mrlt = np.mean(rlts_arr[i], axis=0)
    fancy_plot(mrlt, colors[i])




img_path = "save/{}_{}_{}_{}_{}#{}#{}.png".format(data_name, tp, _noise_num, _kappa, rep, round(geom_scores[1], 5), round(geom_scores[2], 5))
plt.savefig(img_path)




