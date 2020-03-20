import time
import numpy as np


dims = [2,4,8,16]
init_sizes = [0,100,10000]
addings = [0,10,100,1000]
rk_fracs = [0.1,0.5,0.9]

TIME = np.zeros((len(dims)*len(init_sizes)*len(addings)*len(rk_fracs),10))
row = -1

for dim = dims:
    for init = init_sizes:
        # Initialize X (initial points)
        for adding = addings:
            # Initialize Y (points being added)
            for rk_frac = rk_fracs:
                row += 1

                TIME[row,0] = method1(X, Y, rk_frac*diam(X))
                TIME[row,0] = method2(X, Y, rk_frac*diam(X))
                TIME[row,0] = method3(X, Y, rk_frac*diam(X))
                TIME[row,0] = method4(X, Y, rk_frac*diam(X))
                TIME[row,0] = method5(X, Y, rk_frac*diam(X))

                
                for pt in Y: 
                    INFO = method1(X, pt, rk_frac*diam(X),INFO)

                for pt in Y: 
                    INFO = method2(X, pt, rk_frac*diam(X),INFO)

                for pt in Y: 
                    INFO = method3(X, pt, rk_frac*diam(X),INFO)

                for pt in Y: 
                    INFO = method4(X, pt, rk_frac*diam(X),INFO)

                for pt in Y: 
                    INFO = method5(X, pt, rk_frac*diam(X),INFO)
                
