import numpy as np
import time
import kdtree
import mtree
from first_attempt import six_hump_camel_func

def main():
   dim = (2,4,8,16)
   init_sizes = (10,100,1000,10000)
   seed = 1
   test = 0
   method = "mtree"
   print("Initializing with {0}".format(method))
   for n in dim:
      #populating initial data set X
      for num_pts in init_sizes:
         H = np.zeros(num_pts,dtype=[
            ('x',float,n),            # the point 
            ('f',float),              # its value
            ('pt_id',int),            # its index
            ('best_dist',float),      # better neighbor
            #('known',bool)           # is the point known?
           ])
         np.random.seed(seed)
         H['x'] = np.random.uniform(-2,2,(num_pts,n))
         H['f'] = np.apply_along_axis(six_hump_camel_func,1,H['x'])
         H['pt_id'] = np.arange(0,num_pts)
         H['best_dist'] = np.inf
         H = np.sort(H, order = 'f')           #sorted according to function value
         
         a = tuple(map(tuple, np.array(H['x'])))  #pre-processing for mtree constructor
         tree = mtree.MTree()
         tree.add(a[0])

         #a = np.array(H['x']).tolist()         #pre-processing for kdtree constructor
         #tree = kdtree.create(a[0:1],n)

         start = time.time()
         for i in range(1,num_pts):                  
            #vect = tree.search_nn(a[i])[0].data
            #H[i]['best_dist'] = np.linalg.norm(H['x'][i]-vect)
            #tree.add(a[i])
            vect = tree.get_nearest(a[i], limit = 1)
            H[i]['best_dist'] = next(vect)[1]
            tree.add(a[i])
         end = time.time() - start
         print('{2}, {0}, {1}'.format(num_pts, end, n))
         
if __name__ == "__main__":
   main()
