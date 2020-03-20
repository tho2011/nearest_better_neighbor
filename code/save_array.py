import numpy as np
from scipy import spatial
from first_attempt import price01
import numpy as np
import time
import kdtree
#from sklearn.neighbors import BallTree
#from sklearn.neighbors import KDTree
from scipy import spatial


def main():
    num_pts = 10000
    #num_pts_to_add = 1000
    #total_pts_to_add = num_pts + num_pts_to_add
    n = 7
    r = 0.9

    H = np.zeros(num_pts,dtype=[
                                ('x',float,n),            # the point 
                                ('f',float),              # its value
                                ('pt_id',int),            # its index
                                ('best_dist', float)      # dist to better nb
                               ])

    np.random.seed(1)
    H['x'] = np.random.uniform(0,1,(num_pts,n))
    H['f'] = np.apply_along_axis(price01,1,H['x'])
    H['pt_id'] = np.arange(0,num_pts)
    H['best_dist'] = np.inf
    A = np.sort(H, order = 'f')           #sorted according to function value
    a = np.array(A['x']).tolist()         #pre-processing for kdtree constructor
    tree = kdtree.create(a[0:1],n)
    #H = np.append(H,np.zeros(num_pts_to_add,dtype=H.dtype))
    '''
    for i in np.arange(len(H)-num_pts_to_add,len(H)):
       H['x'][i] = np.random.uniform(0,1,n)
       H['pt_id'][i] = i
       H['best_dist'] = np.inf
    H['f'] = np.apply_along_axis(price01,1,H['x'])
    '''     
    #start = time.time()
    #T = spatial.cKDTree(H['x'])
    #end = time.time() - start
    start = time.time()
    #print("Time to initialize tree with {0} pts is {1}".format(total_pts_to_add,end))
    for i in range(1,num_pts):
        vect = tree.search_nn(a[i])[0].data
        dist = np.linalg.norm(A['x'][i]-vect)
        if dist < r:
           H['best_dist'][A['pt_id'][i]] = dist
        tree.add(a[i])
    end = time.time() - start
    print('(dim = {0})(Time to initialize {1} pts, r = {3}, is {2})'.format(n, num_pts, end, r))
    #np.save('data_dim2_seed1',H)
    #print(H)
    #print("Checking")
    #print(H['best_dist'])             

if __name__ == "__main__":
   main()
