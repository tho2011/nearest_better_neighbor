import numpy as np
from scipy.spatial.distance import cdist
import time

def update_nearest_better_neighbor(H):
    """
    Update distances for any new points that have been evaluated
    """
    new_inds = np.where(~H['known'])[0]
    H['known'][new_inds] = True # These points are now known 

    # Loop over new returned points and update their distances
    for new_ind in new_inds:
        dist_to_all = cdist(np.atleast_2d(H['x'][new_ind]).copy(), H['x'], 'euclidean').flatten()
        new_better_than = H['f'][new_ind] < H['f']

        # Update any other points if new_ind is closer and better
        inds = np.logical_and(dist_to_all < H['dist_to_better'], new_better_than)
        H['dist_to_better'][inds] = dist_to_all[inds]
        H['ind_of_better'][inds] = new_ind

        # Since we allow equality when deciding better_than_new and
        # we have to prevent new_ind from being its own better point.
        better_than_new = np.logical_and.reduce((~new_better_than, H['pt_id'] != new_ind))

        # Who is closest to ind and better 
        if np.any(better_than_new):
            ind = dist_to_all[better_than_new].argmin()
            H['ind_of_better'][new_ind] = H['pt_id'][np.nonzero(better_than_new)[0][ind]]
            H['dist_to_better'][new_ind] = dist_to_all[better_than_new][ind]

    return H


# http://infinity77.net/global_optimization/test_functions_nd_P.html
def price01(x):
    return np.sum((np.abs(x) - 5)**2)

def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2;
    term2 = x1*x2;
    term3 = (-4+4*x2**2) * x2**2;

    return  term1 + term2 + term3

if __name__ == "__main__":
    sizes = 2**np.arange(14,15)
    test_times = np.zeros(len(sizes))
    for test_inst, num_pts in enumerate(sizes):
        print(str(test_inst+1) + ' of ' + str(len(sizes)))
        # Generate points in the unit cube
        n = 4

        # Numpy structured array to hold everything (for now)
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
        H['x'] = np.random.uniform(-2,2,(num_pts,n))
        H['f'] = np.apply_along_axis(price01,1,H['x'])
        H['pt_id'] = np.arange(0,num_pts) 
        H['dist_to_better'] = np.inf
        H['ind_of_better'] = -1

        # Intialize other information
        start = time.time()
        H = update_nearest_better_neighbor(H)
        print(H[np.argsort(H['f'])])
        test_times[test_inst] = time.time() - start

    print("The times for the sizes: ", sizes, " is ", test_times)


    # Time how long it takes to append more points
    num_pts_to_add = len(H)
    H = np.append(H,np.zeros(num_pts_to_add,dtype=H.dtype))
    start = time.time()
    for i in np.arange(len(H)-num_pts_to_add,len(H)):
        H['x'][i] = np.random.uniform(0,1,n)
        H['f'][i] = np.random.uniform(0,1)
        H['pt_id'][i] = i
        H['dist_to_better'][i] = np.inf
        H['ind_of_better'][i] = -1

        H = update_nearest_better_neighbor(H)

    print("The time required to add " + str(num_pts_to_add) + ", resulting in " + str(len(H)) + " points is ", time.time()-start)
