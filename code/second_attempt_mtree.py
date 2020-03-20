import numpy as np
import time
import mtree
from first_attempt import six_hump_camel_func

# Generate points in the unit cube
n = 3000
num_pts = 200
seed = 1
#min_n_c = 2
#max_n_c = None
# Numpy structured array to hold everything (for now)
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

H = np.sort(H, order = 'f')
a = tuple(map(tuple, np.array(H['x'])))  #pre-processing for mtree constructor
tree = mtree.MTree() #min_node_capacity=2, max_node_capacity=9
tree.add(a[0])
search_times_mtree = np.zeros(num_pts, dtype=[('i', int),('time',float)])      
print("mtree")
#start = time.time()
for i in range(1,num_pts):
     vect = tree.get_nearest(a[i], limit = 1)
     start = time.time()
     H[i]['best_dist'] = next(vect)[1]
     end = time.time() - start
     search_times_mtree[i] = (i,end)
     tree.add(a[i])
#end = time.time() - start
print(search_times_mtree)
#print(H)
#print('Node capacities: min = {0}, max = {1}'.format(min_n_c,max_n_c))
#print('(mtree, dim = {2})Initializing {0} points in {1} seconds'.format(num_pts, end, n))

#print('(mtree, dim = {2})Building {0} points in {1} seconds'.format(num_pts, end, n))

'''
start = time.time()
vect = tree.get_nearest((0,0,0,0), limit = 1)
print(next(vect))
end = time.time() - start
print("Retrieval time: {0}".format(end))
'''
