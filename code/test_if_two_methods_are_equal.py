import numpy as np
import time
import kdtree
from first_attempt import update_nearest_better_neighbor

# Generate points in the unit cube
n = 4
num_pts = 2**14
# Numpy structured array to hold everything (for now)
H1 = np.zeros(num_pts,dtype=[
            ('x',float,n),            # the point 
            ('f',float),              # its value
            ('pt_id',int),            # its index
            ('best_nb',float,n),      # better neighbor
            ('ind_of_better',int),    # index of better
            ('dist_to_better',float), # index of better
            ('known',bool)            # is the point known?
           ])

np.random.seed(1)
H1['x'] = np.random.uniform(0,1,(num_pts,n))
H1['f'] = np.random.uniform(-10,10,num_pts)
H1['dist_to_better'] = np.inf
H1['ind_of_better'] = -1
H1['best_nb'][0] = None
H1 = np.sort(H1, order = 'f')           #sorted according to function value
H1['pt_id'] = np.arange(0,num_pts)

start = time.time()
H2 = np.copy(H1)
a = np.array(H1['x']).tolist()         #pre-processing for kdtree constructor
tree = kdtree.create(a[0:1],n)        
for i in range(1,num_pts):
    H1[i]['best_nb'] = tree.search_nn(a[i])[0].data
    tree.add(a[i])

for i in range(1,num_pts):
    H1['ind_of_better'][i] = H1['pt_id'][np.where((H1['x'] == H1['best_nb'][i]).all(axis=1))[0][0]]

end = time.time() - start
print('Time for new method with ', num_pts,' is: ',end)

start = time.time()
H2 = update_nearest_better_neighbor(H2)
end = time.time() - start
print('Time for brute force with ', num_pts,' is: ',end)

assert np.array_equal(H1['ind_of_better'],H2['ind_of_better']), 'These should be equal!'
