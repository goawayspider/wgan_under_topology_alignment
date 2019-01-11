import numpy as np
from utils import *

class Simplex():
	def __init__(self,point_list,filtration):
		self.point_list = point_list
		self.filtration = filtration
		self.dim = len(self.point_list)-1

	def sample(self):
		dir_para = [1]*(self.dim+1)
		dir_var = np.random.dirichlet(dir_para, 1).reshape((self.dim+1,1))
		return np.sum(self.point_list * dir_var,axis=0)


class Point():
	def __init__(self):
		self.dim = 0 #the dimension of the highest dimension simplex the point belongs to
		self.simplexs = []#the indexes of simplexs
	def sample(self):
		return np.random.choice(self.simplexs,1)[0]



class Gene_noise():
	def __init__(self,data,alpha=0.1,skeleton_dimension=2,land_mark_num=None):

		if land_mark_num is None:
			self.land_mark_num = data.shape[0]/2
		else:
			self.land_mark_num = land_mark_num
		self.L = random_landmarks(data,self.land_mark_num)
		W = data

		lmrk_tab, _ = lmrk_table(W, self.L)

		wc = gudhi.WitnessComplex(lmrk_tab)
		st = wc.create_simplex_tree(max_alpha_square=alpha, limit_dimension=skeleton_dimension)
		skeleton = st.get_skeleton(skeleton_dimension)
		self.simplexs = []
		self.points = [Point() for _ in range(self.land_mark_num)]
		self.graph = np.zeros((self.land_mark_num,self.land_mark_num),dtype=np.float32)

		simplex_id = -1
		for i in range(len(skeleton)):
			dim = len(skeleton[i][0])-1
			if dim > 0:
				simplex_id += 1
				self.simplexs.append(Simplex(self.L[skeleton[i][0]],skeleton[i][1]))
				if dim == 1:
					self.graph[skeleton[i][0][0],skeleton[i][0][1]]=1
					self.graph[skeleton[i][0][1],skeleton[i][0][0]]=1
				for j in range(dim+1):
					point_id = skeleton[i][0][j]
					if self.simplexs[simplex_id].dim == self.points[point_id].dim:
						self.points[point_id].simplexs.append(simplex_id)
					elif self.simplexs[simplex_id].dim > self.points[point_id].dim:
						self.points[point_id].dim = self.simplexs[simplex_id].dim
						del self.points[point_id].simplexs[:]
						self.points[point_id].simplexs.append(simplex_id)

		self.graph = self.graph / np.sum(self.graph,axis=-1)[:,np.newaxis]
		#print(self.graph)


	def get_noise(self,noise_num):
		noise = []
		point_id = np.random.choice(self.land_mark_num,1)[0]
		for i in range(noise_num):
			simplex_id = self.points[point_id].sample()
			noise.append(self.simplexs[simplex_id].sample())

			point_id = np.random.choice(self.land_mark_num,1,True,self.graph[point_id])[0]

		return noise




def main():
	X = np.random.normal(0.0,1.0,size=(100,2))

	g = Gene_noise(X)
	print(g.get_noise(10))

if __name__ == '__main__':
	main()

