import numpy as np
from scipy import spatial
import copy
from scipy.sparse import lil_matrix, hstack, vstack
from first_attempt import price01
import numpy as np
import time
import kdtree
#from sklearn.neighbors import KDTree

def main():

    num_pts = 10000
    num_pts_to_add = 1000
    total_pts_to_add = num_pts + num_pts_to_add
    n = 4
    rad = 0.5

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
    T = spatial.KDTree(H['x'])

    for i in range(1,num_pts):
        vect = tree.search_nn(a[i])[0].data
        dist = np.linalg.norm(A['x'][A['pt_id'][i]]-vect)
        if dist < rad:
           H['best_dist'][i] = dist
        tree.add(a[i])
    #print("Checking")
    #print(H['best_dist'])

    start = time.time()
    for i in np.arange(num_pts, total_pts_to_add):
       #l = T.query_radius(H['x'][i].reshape(1,-1),r = rad)
       l = T.query_ball_point(H['x'][i],rad)
       dist_all = spatial.distance.cdist(H['x'][[i]], H['x'][l], 'euclidean')[0]
       #dist_all = spatial.distance.cdist(H['x'][[i]], H['x'][l[0]], 'euclidean')[0]
       for count,j in enumerate(l):
       #for count,j in enumerate(l[0]):
          if j == i:
             continue
          # dist_ij = np.linalg.norm(H['x'][i]-H['x'][j])
          dist_ij = dist_all[count]
          if H['f'][i] > H['f'][j]: #finding better neighbor for pt to be updated
             if dist_ij < H['best_dist'][i]:
                H['best_dist'][i] = dist_ij 
          else:
             if dist_ij < H['best_dist'][j]:
                H['best_dist'][j] = dist_ij
    end = time.time() - start

    #print(H['best_dist'])
    print('(sklearn kd tree range)(dim = {3})Time for adding {0} points with initial {1} pts is: {2}'.format(num_pts_to_add, num_pts, end, n))
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
