import Selling
import numpy as np
from LinearP import *

dim=3
bounds = (3,)
case = (1,)
A = np.random.standard_normal( (dim,dim,)+bounds )
a = A[:,:,case]

m = dotP_AA(transP(a),a)
M = dotP_AA(transP(A),A)

coefs,offsets = Selling.DecompP(m)
Coefs,Offsets = Selling.DecompP(M)

# Check reconstruction of m and M
rec = multP(coefs,outerP(offsets,offsets)).sum(2)
Rec = multP(Coefs,outerP(Offsets,Offsets)).sum(2)


aI = inverseP(a)
AI = inverseP(A)
print(dotP_AA(A,AI))
