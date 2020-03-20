import numpy as np
import time
import kdtree
from first_attempt import six_hump_camel_func

# Generate points in the unit cube
n = 16
num_pts = 1000
seed = 1
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

H = np.sort(H, order = 'f')           #sorted according to function value
a = np.array(H['x']).tolist()         #pre-processing for kdtree constructor
tree = kdtree.create(a[0:1],n)
#search_times_kdtree = np.zeros(num_pts, dtype=[('i', int),('time',float)])      
start = time.time()
for i in range(1,num_pts):
    #start = time.time()
    vect = tree.search_nn(a[i])[0].data
    #end = time.time() - start
    H[i]['best_dist'] = np.linalg.norm(H['x'][i]-vect)
    #search_times_kdtree[i] = (i,end)
    tree.add(a[i])
end = time.time() - start
#print(search_times_kdtree)
#print(H)
print('(kdtree, dim = {2})Initializing {0} points in {1} seconds'.format(num_pts, end, n))

#print('(kdtree, dim = {2})Building {0} points in {1} seconds'.format(num_pts, end, n))

'''
start = time.time()
print(tree.search_nn((0,0,0,0))[0].data)
end = time.time() - start
print("Retrieval time: {0}".format(end))
'''
