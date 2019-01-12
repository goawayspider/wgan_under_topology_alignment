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



def reduce_dim(data,noise_dim):
    images,_ = data

    np.random.shuffle(images)
    images = images[:3000]
 
    #emb image_num * noise_dim
    emb = manifold.Isomap(n_neighbors=20, n_components=noise_dim).fit_transform(images)
    #emb,err = manifold.locally_linear_embedding(images,n_neighbors=30,n_components=noise_dim)

    #print("Done. Reconstruction error: %g" % err)
    print('done')
    emb = emb.astype('float32')

    return images,emb



with gzip.open('save/mnist.pkl.gz', 'rb') as f:
	train_data, dev_data, test_data = pickle.load(f)

image,emb = reduce_dim(train_data,20)
noise = Gene_noise(emb).get_noise()
gaussi_noise = np.random.normal(0,1,(3000,20))


rlts1 = rlts(image)
rlts2 = rlts(noise)
rlts3 = rlts(gaussi_noise)


print(geom_score(rlts1,rlts2))
print(geom_score(rlts1,rlts3))

mrlt1 = np.mean(rlts1, axis=0)
fancy_plot(mrlt1)
mrlt2 = np.mean(rlts2, axis=0)
fancy_plot(mrlt2,'r')
mrlt3 = np.mean(rlts3, axis=0)
fancy_plot(mrlt3,'y')

plt.savefig("save/aaa.png")




