import numpy as np
from scipy import spatial
import copy
from first_attempt import price01
import numpy as np
import time
import kdtree

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

    num_pts = 100000
    n = 4
    r = 0.5
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
    a = np.array(A['x']).tolist()         #pre-processing for kdtree constructor
    tree = kdtree.create(a[0:1],n)
    
    i = 0
    j = num_pts - 2
    
    R[num_pts - 1] = (A['pt_id'][0], np.inf)
    start = time.time()
    for k in range(1,num_pts):
        vect = tree.search_nn(a[k])[0].data
        dist = np.linalg.norm(A['x'][k]-vect)
        if dist < r:
           R[i] = (k, dist)
           i+=1
        else:
           R[j] = (k, np.inf)
           j-=1 
        tree.add(a[k])    
    I = i
    end = time.time() - start
    print('Initializing {0} points with r = {1} and seed = {2} in {3} seconds'.format(num_pts, r, seed, end))
    #print(R)
    print()
    start = time.time()
    r = 0.10
    R, I = restrict(R,r,I)
    end = time.time() - start
    print("Time to restrict {0} pts (r = {1}) is {2} seconds".format(num_pts,r,end))
    #print("Restricting to r =",r)
    #print(R)
    print()
    #print("Retrieving points without rk better neighbor")
    #print(retrieve(R,I))


    
if __name__ == "__main__":
    main()
