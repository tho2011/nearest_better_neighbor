import numpy as np
import copy
from scipy.sparse import lil_matrix, hstack, vstack
import scipy as sp
from scipy import spatial
from first_attempt import price01
import time

def update(A, H, i, r):
   too_far = np.ones(i, dtype = bool)
   for j in range(i-1):
      if too_far[j]:
         dist = np.linalg.norm(H['x'][i]-H['x'][j])
         if dist < r:
            A[j,i] = dist
            A[i,j] = dist
         if dist > 2*r:
            too_far[A.rows[j]] = False
   #    else:
   #       count+=1
   # print("Distance calculations saved at i =",i," is ", count)
   return A

def update1(A, H, i, r):

   dist = sp.spatial.distance.cdist(H['x'][[i]],H['x'][:i]).flatten()
   inds = dist<r
   A[i,inds] = dist[inds]
   for j in range(len(inds)):
       if inds[j]:
           A[j,i] = dist[j]
   return A

def restrict(A,r):
   s = len(A.nonzero()[0])
   #print("s =",s)
   i = copy.deepcopy(A.nonzero()[0])
   j = copy.deepcopy(A.nonzero()[1])
   #print("i list = ",i)
   #print("j list = ",j)
   for k in range(s):
      if A[i[k],j[k]] > r:
         A[i[k],j[k]] = 0
   return A

def main():
    num_pts = 10000
    n = 4
    A = lil_matrix(num_pts,num_pts)
    r = 0.4

    H = np.zeros(num_pts,dtype=[
                                ('x',float,n),            # the point 
                                ('f',float),              # its value
                                ('pt_id',int),            # its index
                               ])

    np.random.seed(1)
    H['x'] = np.random.uniform(0,1,(num_pts,n))
    H['f'] = np.apply_along_axis(price01,1,H['x'])
    H['pt_id'] = np.arange(0,num_pts)

    start = time.time()
    D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(H['x'], 'euclidean'))
    D[D>r] = 0
    A = lil_matrix(D)

    num_pts_to_add = 1000
    H = np.append(H,np.zeros(num_pts_to_add,dtype=H.dtype))

    v = lil_matrix((num_pts_to_add+num_pts,num_pts_to_add))
    h = lil_matrix((num_pts_to_add,num_pts))
    A = vstack([A,h],format='lil')
    A = hstack([A,v],format='lil')

    for i in np.arange(len(H)-num_pts_to_add,len(H)):
       H['x'][i] = np.random.uniform(0,1,n)
       H['f'][i] = np.random.uniform(0,1)
       H['pt_id'][i] = i
       A = update1(A, H, i, r)
    end = time.time()-start
    
    print("(Sparse matrix)(dim = {3}, r = {4})Time to update {0} pts with initial {1} pts is {2}".format(num_pts_to_add, num_pts, end, n, r))
    #A = restrict(A,0.3)
    #print(A)



if __name__ == "__main__":
    main()
