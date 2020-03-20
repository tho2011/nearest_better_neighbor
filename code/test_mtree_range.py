import numpy as np
from scipy import spatial
import copy
from scipy.sparse import lil_matrix, hstack, vstack
from first_attempt import price01
import numpy as np
import time
import kdtree
import mtree

def main():

    num_pts = 10000
    num_pts_to_add = 1000
    total_pts_to_add = num_pts + num_pts_to_add
    n = 4
    r = 0.5

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
    A = np.sort(H, order = 'f')             #sorted according to function value
    a = np.array(A['x']).tolist()           #pre-processing for kdtree constructor
    #b = tuple(map(tuple, np.array(A['x']))) #pre-processing for mtree constructor
    T = kdtree.create(a[0:1],n)
    M = mtree.MTree()
    H = np.append(H,np.zeros(num_pts_to_add,dtype=H.dtype))
    for i in np.arange(len(H)-num_pts_to_add,len(H)):
       H['x'][i] = np.random.uniform(0,1,n)
       H['pt_id'][i] = i
       H['best_dist'] = np.inf
    H['f'] = np.apply_along_axis(price01,1,H['x'])
    t = tuple(map(tuple, np.array(H['x'])))
    dict_index = {}
    for i in range(len(H)):
        dict_index[t[i]] = i      

    for i in range(1,num_pts):
        vect = T.search_nn(a[i])[0].data
        dist = np.linalg.norm(A['x'][i]-vect)
        if dist < r:
           H['best_dist'][i] = dist
        T.add(a[i])
  
    #print("Checking")
    #print(H['best_dist'])
    for i in range(0,total_pts_to_add):
       M.add(tuple(H['x'][i]))
    start = time.time()
    for i in np.arange(num_pts, total_pts_to_add):
       #start = time.time()
       l = M.get_nearest(tuple(H['x'][i]),rad=r)
       for k in l:
          ind = dict_index[k[0]]
          if H['f'][i] > H['f'][ind]: #finding better neighbor for pt to be updated
             if k[1] < H['best_dist'][i]:
                H['best_dist'][i] = k[1] 
          else:
             if k[1] < H['best_dist'][ind]:
                H['best_dist'][ind] = k[1]
       #M.add(tuple(H['x'][i]))
       #end = time.time() - start
       #print('Time to update iteration {0} is {1}'.format(i, end))
    end = time.time() - start

    #print(H['best_dist'])
    print('(M tree range python3, dim = {3}, r = {4})Time for adding {0} points with initial {1} pts is: {2}'.format(num_pts_to_add, num_pts, end, n, r))
    #print(H['x'])


if __name__ == "__main__":
    main()
