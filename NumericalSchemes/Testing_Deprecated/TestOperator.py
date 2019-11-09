import numpy as np
import scipy.sparse as sp

import sys; sys.path.append("../..") # Allow import from parent directory
from NumericalSchemes import LinearPDE
from NumericalSchemes import LinearParallel as lp

dim=2
bounds = (3,4)

diff = np.zeros((dim,dim,)+bounds)
diff[0,0]=1
diff[1,1]=1

omega = np.zeros((dim,)+bounds)
omega[1]=1

mult = np.ones(bounds)

coef,(row,col) = LinearPDE.OperatorMatrix(diff,bc='Neumann')
lap = sp.coo_matrix((coef,(row,col)))
print(lap.toarray().astype(int))


coef,(row,col) = LinearPDE.OperatorMatrix(diff,divergenceForm=True)
lap2 = sp.coo_matrix((coef,(row,col)))

coef,(row,col) = LinearPDE.OperatorMatrix(diff,omega,mult)

lap3=sp.coo_matrix((coef,(row,col)))
print((lap3-lap2).toarray())