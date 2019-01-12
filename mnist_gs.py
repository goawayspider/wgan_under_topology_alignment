import matplotlib as mpl
mpl.use('Agg')
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
store = True
store_path = "mnist_{}.npy"

def reduce_dim(data,noise_dim):
    images,_ = data

    np.random.shuffle(images)
    images = images[:6000]
 
    #emb image_num * noise_dim
    emb = manifold.Isomap(n_neighbors=5, n_components=noise_dim, n_jobs = 10).fit_transform(images)
    #emb,err = manifold.locally_linear_embedding(images,n_neighbors=30,n_components=noise_dim)

    #print("Done. Reconstruction error: %g" % err)
    print('done')
    emb = emb.astype('float32')

    return images,emb

_noise_num = 10000
_kappa = 0.5 # to control the threshold to construct the cech complex
tp = datetime.datetime.now().isoformat()

img_path = "save/{}_{}_{}.png".format(tp, _noise_num, _kappa)




with gzip.open('save/mnist.pkl.gz', 'rb') as f:
	train_data, dev_data, test_data = pickle.load(f)

if not store:
    image,emb = reduce_dim(train_data,20)
    np.save(store_path.format("image"), image)
    np.save(store_path.format("emb"), emb)
else:
    image = np.load(store_path.format("image"))
    emb = np.load(store_path.format("emb"))
    
noise = Gene_noise(emb,kappa = _kappa).get_noise(noise_num = _noise_num)
gaussi_noise = np.random.normal(0,1,(3000,20))

def store_rlts(data, arr, index, n_job):
    out = rlts_pooled(data, n = 100, n_jobs = n_job)
    arr[index] = out
    


data_parts = [image, emb, noise, gaussi_noise]
names = ["raw", "mfd", "noise", "gaussian"]
TOTAL = 100
n_jobs= [0.2, 0.2, 0.5, 0.1]
rlts_arr = [None] * len(data_parts)

threads = [None]* len(data_parts)

for i in range(len(data_parts)):
    t = threading.Thread(target = store_rlts, name = names[i], args = (data_parts[i], rlts_arr, i, int(n_jobs[i]*TOTAL)))
    threads[i] = t
    t.start()


# wait
for i in range(len(data_parts)):
    threads[i].join()



for i in range(1, len(data_parts)):
    print("{}=={} = {}".format(names[0], names[i] ,geom_score(rlts_arr[0], rlts_arr[i])))

colors = ['b', 'r', 'y', 'c']

for i in range(len(data_parts)):
    mrlt = np.mean(rlts_arr[i], axis=0)
    fancy_plot(mrlt, colors[i])

plt.savefig(img_path)




