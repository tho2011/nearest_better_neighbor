import numpy as np
from scipy import spatial
import copy
from first_attempt import price01
import numpy as np
import time
import kdtree
import mtree

def restrict(R,r,I):
    m = I-1
    for i in range(I):
         if R['best_dist'][I-i-1] > r:
             #R['best_dist'][I-i-1] = np.inf
             swap1 = copy.deepcopy(R[m])
             swap2 = copy.deepcopy(R[I-i-1])
             R[m] = swap2
             R[I-i-1] = swap1
             R['best_dist'][m] = np.inf            
             m-=1
    return R, m+1
   
def retrieve(R,I):
    return R['pt_id'][I:]

def main():

    num_pts = 1000
    n = 4
    r = 0.9
    seed = 1
    R = np.zeros(num_pts,dtype=[
                                ('pt_id',int),            # its index
                                ('best_dist', float)      # dist to better nb
                               ])
    H = np.zeros(num_pts,dtype=[
                                ('x',float,n),            # the point 
                                ('f',float),              # its value
                                ('pt_id',int),            # its index
                               ])

    np.random.seed(seed)
    H['x'] = np.random.uniform(0,1,(num_pts,n))
    H['f'] = np.apply_along_axis(price01,1,H['x'])
    H['pt_id'] = np.arange(0,num_pts)
    start = time.time()
    A = np.sort(H, order = 'f')           #sorted according to function value
    end = time.time() - start
    print('Time to pre-sort points for tree input = {0} seconds'.format(end))
    b = np.array(A['x']).tolist() 
    a = tuple(map(tuple, np.array(A['x'])))  #pre-processing for mtree constructor
    tree = mtree.MTree(min_node_capacity=2, max_node_capacity=8)
    tree1 = kdtree.create(b[0:1],n)
    #tree.add(a[0])
    i = 0
    j = num_pts - 2
    
    R[num_pts - 1] = (A['pt_id'][0], np.inf)
    start = time.time()
    for k in range(1,num_pts):
        #vect = tree.search_nn(a[k])[0].data
        vect = tree.get_nearest(a[k], limit = 1)
        try:
           R[i] = (k, next(vect)[1])
           i+=1
        except StopIteration:
           R[j] = (k, np.inf)
           j-=1 
        tree.add(a[k])
    I = i
    end = time.time() - start
    print('(mtree)Initializing {0} points with r = {1} and seed = {2} in {3} seconds'.format(num_pts, r, seed, end))
    #print(R)
    print()
    start = time.time()
    r = 0.10
    R, I = restrict(R,r,I)
    end = time.time() - start
    #print("Time to restrict {0} pts (r = {1}) is {2} seconds".format(num_pts,r,end))
    print()
    I = tree.get_nearest((0,0,0,0), rad = 0.6)
    start = time.time()
    #for i in I:
     #   print(i[1])
    #end = time.time() - start
    #print("Time to perform a range search is {0} seconds".format(end))
    
if __name__ == "__main__":
    main()
