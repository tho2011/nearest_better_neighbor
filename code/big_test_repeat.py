import numpy as np
from scipy import spatial
from first_attempt import price01
import numpy as np
import time
import kdtree
#from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree

def main():
   dim = (2,4,8,16)
   init_sizes = (0,100,10000)
   addings = (0,10,100,1000)
   radii = np.array((0.1,0.5,0.9))
   seed = 1
   test = 0
   data_structure = "spatial kd-tree"
   c = 0
   files = ["data_dim2_seed1.npy","data_dim4_seed1.npy","data_dim8_seed1.npy","data_dim16_seed1.npy"]
   for n in dim:
      #populating initial data set X
      for num_pts in init_sizes:
         if num_pts == 10000:
            print "load pre-initialized data if initializing 10,000 pts for any dimension"
            X = np.load(files[c])
            c += 1
         else:
            X = np.zeros(num_pts,dtype=[
                             ('x',float,n),            # the point 
                             ('f',float),              # its value
                             ('pt_id',int),            # its index
                             ('best_dist', float)      # dist to better nb
                            ])
            np.random.seed(seed)
            X['x'] = np.random.uniform(0,1,(num_pts,n))
            if num_pts != 0:
               X['f'] = np.apply_along_axis(price01,1,X['x'])
            X['pt_id'] = np.arange(0,num_pts)
            X['best_dist'] = np.inf
            A = np.sort(X, order = 'f')           #sorted according to function value
            a = np.array(A['x']).tolist()         #pre-processing for kdtree constructor
            #initializing best distance array
            #start = time.time()
            if num_pts != 0:
               tree = kdtree.create(a[0:1],n)
               for i in range(1,num_pts):                  
                  vect = tree.search_nn(a[i])[0].data
                  dist = np.linalg.norm(A['x'][i]-vect)
                  if dist < radii[len(radii)-1]:
                     X['best_dist'][A['pt_id'][i]] = dist
                  tree.add(a[i])
            #end = time.time() - start
            #print('(dim = {})Time to initialize {} pts is {}'.format(n, num_pts, end))
         #populating new data set Y
         for num_pts_to_add in addings:
            total_pts_to_add = num_pts + num_pts_to_add
            Y = np.zeros(num_pts_to_add,dtype=X.dtype)
            for i in np.arange(len(Y)):
               Y['x'][i] = np.random.uniform(0,1,n)
               Y['pt_id'][i] = i + num_pts
            Y['best_dist'] = np.inf
            if num_pts_to_add != 0:
               Y['f'] = np.apply_along_axis(price01,1,Y['x'])
            for rad in radii:
               #rad = rad * np.sqrt(n)
               if num_pts == 0 and num_pts_to_add == 0:
                  print('Nothing to initialize and add in dim = {0}, r = {1}'.format(n, rad))
                  continue
               H = np.append(X,Y)                          #re-defining H every radius for consistency
               if rad < radii[len(radii)-1]:
                  for i in range(num_pts):
                     if H['best_dist'][i] > rad:
                        H['best_dist'][i] = np.inf
               start = time.time()
               #T = BallTree(H['x'])
               #T = KDTree(H['x'])                  #sklearn kd-tree
               T = spatial.KDTree(H['x'])
               #T = spatial.cKDTree(H['x'])
               #updating best distance array 
               for i in np.arange(num_pts, total_pts_to_add):
                  #l = T.query_radius(H['x'][i].reshape(1,-1),r = rad)
                  l = T.query_ball_point(H['x'][i],rad)
                  #dist_all = spatial.distance.cdist(H['x'][[i]], H['x'][l[0]], 'euclidean')[0]
                  dist_all = spatial.distance.cdist(H['x'][[i]], H['x'][l], 'euclidean')[0]
                  #for count,j in enumerate(l[0]):
                  for count,j in enumerate(l):
                     if j == i:
                        continue
                     dist_ij = dist_all[count]
                     if H['f'][i] > H['f'][j]: #finding better neighbor for pt to be updated
                        if dist_ij < H['best_dist'][i]:
                           H['best_dist'][i] = dist_ij 
                     else:
                        if dist_ij < H['best_dist'][j]:
                           H['best_dist'][j] = dist_ij
               end = time.time() - start
               test += 1
               print('{0}, {1}, dim = {2}, #initial_pts = {3}, #new_pts = {4}, rad = {5}, time = {6}'.format(test, data_structure, n, num_pts, num_pts_to_add, rad, end))             
     
if __name__ == "__main__":
   main()
