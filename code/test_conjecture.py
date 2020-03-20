'''Python code for tracking changes to NBN digraph'''

import numpy as np
#import time
from first_attempt import update_nearest_better_neighbor

n = 4
num_pts = 10
num_pts_up = 3
H = np.zeros(num_pts,dtype=[
                                    ('x',float,n),            # the point 
                                    ('f',float),              # its value
                                    ('pt_id',int),            # its index
                                    ('dist_to_better',float), # distance to better
                                    ('ind_of_better',int),    # index of better
                                    ('known',bool)            # is the point known?
                                   ])


# Initialize our points and values 
np.random.seed(1)
H['x'] = np.random.uniform(0,1,(num_pts,n))
H['f'] = np.random.uniform(-10,10,num_pts)
H['pt_id'] = np.arange(0,num_pts) 
H['dist_to_better'] = np.inf
H['ind_of_better'] = -1
H = update_nearest_better_neighbor(H)
print('Initial:')
print(H[np.argsort(H['f'])])
print()

for i in range(num_pts_up):
    H = np.append(H, np.zeros(1,dtype=H.dtype))
    H[num_pts+i]['x'] = np.random.uniform(0,1)
    H[num_pts+i]['f'] = np.random.uniform(-10,10)
    H[num_pts+i]['pt_id'] = num_pts+i 
    H[num_pts+i]['dist_to_better'] = np.inf
    H[num_pts+i]['ind_of_better'] = -1
    print('After updating', i+1,'point/s')
    H = update_nearest_better_neighbor(H)
    print(H)
    print()
