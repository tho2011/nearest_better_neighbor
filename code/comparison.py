"""
Compare both methods to ensure outputs match; compare performance as well
"""

import numpy as np
from scipy.spatial.distance import cdist
import time
import kdtree
import copy 
from first_attempt import update_nearest_better_neighbor, six_hump_camel_func

#modify parameters here
num_pts = 10000     #data points
n = 4            #dimension
seed = 26

H = np.zeros(num_pts,dtype=[   #Brute force data
                            ('x',float,n),            # the point 
                            ('f',float),              # its value
                            ('pt_id',int),            # its index
                            ('ind_of_better',int),    # index of better
                            ('dist_to_better',float), # distance to better
                            ('known',bool)            # is the point known?
                            ])

M = np.zeros(num_pts,dtype=[  #Sorted kd data
                            ('x',float,n),            # the point 
                            ('f',float),              # its value
                            ('pt_id',int),            # its index
                            ('best_nb',float,n),      # better neighbor
                            ])

np.random.seed(seed)
#filling brute force data
H['x'] = np.random.uniform(-2,2,(num_pts,n))
H['f'] = np.apply_along_axis(six_hump_camel_func,1,H['x'])
H['pt_id'] = np.arange(0,num_pts) 
H['ind_of_better'] = -1
H['dist_to_better'] = np.inf
#filling sorted kd data
M['x'] = copy.deepcopy(H['x'])
M['f'] = copy.deepcopy(H['f'])
M['best_nb'][0] = None
M['pt_id'] = np.arange(0,num_pts)

start = time.time()
H = update_nearest_better_neighbor(H) #brute force update
end = time.time() - start
print("Brute force time =",end)

start = time.time()
M = np.sort(M, order = 'f')           #sorted kd update
a = np.array(M['x']).tolist()
tree = kdtree.create(a[0:1],n)
for i in range(1,num_pts):
    M[i]['best_nb'] = tree.search_nn(a[i])[0].data
    tree.add(a[i])
end = time.time() - start
print("Sort kd time =",end)
for i in range(1,num_pts):
    for j in range(n):
        assert(H[H[M[i]['pt_id']]['ind_of_better']]['x'][j] == M[i]['best_nb'][j])
print("The two methods have the same output")

'''
print()
np.set_printoptions(precision=4)
print("Brute force: seed =",seed)
print(H)
print()
print("Sort + kdtree: seed =",seed)
print(M)
'''
