import numpy as np 
import gzip
import pickle
from geom_score import *
'''
BATCH_SIZE = 100
train_num =50000
dev_num = 10000
test_num = 10000
gene_num = 


train_gen, dev_gen, test_gen ,noi_gen= tflib.mnist.load(BATCH_SIZE, BATCH_SIZE, 0)
fake_image = np.loadtxt('')

def gene():
	fake_batchs = np.reshpae(fake_image,[-1,BATCH_SIZE,784])
	for i in range(len(fake_batchs)):
		yield fake_batchs[i]
fake_gen = gene()
fake_lable = classifier(BATCH_SIZE,train_gen,test_gen,fake_gen)
'''
with gzip.open('../tflib/tmp/mnist.pkl.gz', 'rb') as f:
	train_data, dev_data, test_data = pickle.load(f,encoding='latin1')
real_image,_ = train_data
gene_image = np.loadtxt('./save/generate3999.txt')

real_image = real_image[:1000]
gene_image = gene_image[:1000]

rlts1 = rlts(real_image)
rlts2 = rlts(gene_image)
print(geom_score(rlts1,rlts2))