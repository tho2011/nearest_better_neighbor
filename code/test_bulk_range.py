import numpy as np
from scipy import spatial
import copy
from scipy.sparse import lil_matrix, hstack, vstack
from first_attempt import price01
import numpy as np
import time
import kdtree
from sklearn.neighbors import BallTree

def main():

    num_pts = 1000
    num_pts_to_add = 10000
    total_pts_to_add = num_pts + num_pts_to_add
    n = 4
    rad = 0.8

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
    H = np.append(H,np.zeros(num_pts_to_add,dtype=H.dtype))
    for i in np.arange(len(H)-num_pts_to_add,len(H)):
       H['x'][i] = np.random.uniform(0,1,n)
       H['pt_id'][i] = i
    H['best_dist'] = np.inf
    H['f'] = np.apply_along_axis(price01,1,H['x'])
    #print("printing appended H")
    #print(H)
    #print()
    B = np.sort(H[num_pts:total_pts_to_add], order = 'f')
    #print("printing B")
    #print(B)
    b = np.array(B['x']).tolist()  
    T = BallTree(H['x'][0:num_pts])
    
    #initial data pts
    for i in range(1,num_pts):
        vect = tree.search_nn(a[i])[0].data
        dist = np.linalg.norm(A['x'][i]-vect)
        if dist < rad:
           H['best_dist'][A['pt_id'][i]] = dist
        #print(A['pt_id'][i], dist)
        tree.add(a[i])
    '''Up to this part is correct....'''
    '''
    print("Checking")
    for i in range(num_pts):
       print(H['pt_id'][i],H['best_dist'][i])
    '''
    #bulk adding new points
    start = time.time()
    query_tree = kdtree.create(b[0:1],n)  #setting up query kd-tree and rooting the first point
    l = T.query_radius(H['x'][B['pt_id'][0]].reshape(1,-1),r = rad)
    dist_all = spatial.distance.cdist(H['x'][[B['pt_id'][0]]], H['x'][l[0]], 'euclidean')[0]
    for count,j in enumerate(l[0]):
        dist_ij = dist_all[count]
        if H['f'][B['pt_id'][0]] > H['f'][j]: #finding better neighbor for pt to be updated
            if dist_ij < H['best_dist'][B['pt_id'][0]]:
                H['best_dist'][B['pt_id'][0]] = dist_ij 
        else:
            if dist_ij < H['best_dist'][j]:
                H['best_dist'][j] = dist_ij
    
    #dealing with rest of query points
    for i in np.arange(num_pts+1, total_pts_to_add):
       vect = query_tree.search_nn(b[i-num_pts])[0].data
       dist = np.linalg.norm(B['x'][i-num_pts]-vect)
       if dist < rad:
           H['best_dist'][B['pt_id'][i-num_pts]] = dist
       query_tree.add(b[i-num_pts])
       l = T.query_radius(H['x'][B['pt_id'][i-num_pts]].reshape(1,-1),r = rad)
       dist_all = spatial.distance.cdist(H['x'][[B['pt_id'][i-num_pts]]], H['x'][l[0]], 'euclidean')[0]
       for count,j in enumerate(l[0]):
          # dist_ij = np.linalg.norm(H['x'][i]-H['x'][j])
          dist_ij = dist_all[count]
          if H['f'][B['pt_id'][i-num_pts]] > H['f'][j]: #finding better neighbor for pt to be updated
             if dist_ij < H['best_dist'][B['pt_id'][i-num_pts]]:
                H['best_dist'][B['pt_id'][i-num_pts]] = dist_ij 
          else:
             if dist_ij < H['best_dist'][j]:
                H['best_dist'][j] = dist_ij
    end = time.time() - start

    #print(H['best_dist'])
    print('(bulk adjusted range)(dim = {3}, rad = {4})Time for adding {0} points with initial {1} pts is: {2}'.format(num_pts_to_add, num_pts, end, n, rad))
    #print(H['x'])
    l = []
    start = time.time()
    for i in H:
       if i['best_dist'] == np.inf:
          l.append(i)
    end = time.time() - start
    print("Retrieval time for total {0} pts is {1}".format(total_pts_to_add, end))

if __name__ == "__main__":
    main()
