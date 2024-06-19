# -*- coding: utf-8 -*-
# doTRG.py
import numpy as np
from numpy import linalg as LA
from ncon import ncon


def  doTRG(Ain,chiM,chiH,chiV, midsteps = 10):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 26/1/2019
------------------------
Implementation of one iteration of TRG, based upon insertion of optimised \
isometries, for coarse-graining a square-lattice tensor network. Input \
'A' is a four index tensor that defines the partition function while \
'chiM', 'chiH' and 'chiV' are the bond dimensions. Output 'Aout' is the \
coarse-grained 4-index tensor, while 'q', 'v' and 'w' are the isometries \
used in the coarse-graining. Normalization factor is given by 'Anorm', \
while vector 'SPerrs' gives the truncation errors at each step of the \
coarse-graining.

Optional arguments:
`midstep::Int=10`: number of iterations in the optimization of isometries
"""

    chiHI = Ain.shape[0]
    chiVI = Ain.shape[1]

    ##### determine 'q' isometry
    chitemp = min(chiM,chiHI*chiVI)
    q = np.random.rand(chiHI*chiVI,chitemp)
    qenv = ncon([Ain,Ain,Ain,Ain,Ain,Ain,Ain,Ain],[[-1,-2,11,12],[7,8,11,9],[5,12,1,2],[5,9,3,4],
        [-3,-4,13,14],[7,8,13,10],[6,14,1,2],[6,10,3,4]]).reshape(chiHI*chiVI,chiHI*chiVI)

    SP1exact = np.trace(qenv)
    for k in range(midsteps):
        ut,st,vht = LA.svd(qenv @ q,full_matrices=False)
        q = ut @ vht
    
    SP1err = (SP1exact - np.trace(q.T @ qenv @ q)) / SP1exact
    q = q.reshape(chiHI,chiVI,chitemp)
    qA = ncon([q,Ain],[[1,2,-3],[1,2,-2,-1]])

    ###### determine 'v' isometry
    chitemp = min(chiH,chiHI*chiHI)
    v = np.random.rand(chiHI*chiHI,chitemp)
    venv = ncon([q,q,qA,qA,q,q,qA,qA,q,q,qA,qA,q,q,qA,qA],[[5,7,15],[5,8,18], [17,-1,18],[17,-2,19],
                 [9,12,19],[9,11,13],[1,4,13],[1,3,15],[6,7,16],[6,8,21],[20,-3,21],[20,-4,22],
                 [10,12,22],[10,11,14],[2,4,14],[2,3,16]]).reshape(chiHI**2,chiHI**2)

    SP2exact = np.trace(venv)
    for k in range(midsteps):
        ut,st,vht = LA.svd(venv @ v, full_matrices=False)
        v = ut @ vht
    
    SP2err = (SP2exact - np.trace(v.T @ venv @ v)) / SP2exact
    v = v.reshape(chiHI,chiHI,chitemp)
    vA = ncon([v,qA,qA],[[2,3,-3],[1,2,-1],[1,3,-2]])

    ###### determine 'w' isometry
    chitemp = min(chiV,chiVI*chiVI)
    w = np.random.rand(chiVI*chiVI,chitemp)
    wenv = ncon([q,q,qA,qA,q,q,qA,qA,q,q,qA,qA,q,q,qA,qA],[[17,-1,18],[17,-2,19],[1,3,19],
                 [1,4,13],[5,8,13],[5,7,15],[10,11,15],[10,12,18],[20,-3,21],[20,-4,22],
                 [2,3,22],[2,4,14],[6,8,14],[6,7,16],[9,11,16],[9,12,21]]).reshape(chiVI**2,chiVI**2)

    SP3exact = np.trace(wenv)
    for k in range(midsteps):
        ut,st,vht = LA.svd(wenv @ w, full_matrices=False)
        w = ut @ vht
    
    SP3err = (SP3exact - np.trace(w.T @ wenv @ w)) / SP3exact
    w = w.reshape(chiVI,chiVI,chitemp)
    wA = ncon([w,q,q],[[2,3,-3],[1,2,-1],[1,3,-2]])

    ##### compute new 'A' tensor
    Atemp = ncon([vA,wA,vA,wA],[[3,1,-1],[1,4,-2],[4,2,-3],[2,3,-4]])
    Alocnorm = LA.norm(Atemp.flatten())

    Aout = Atemp / Alocnorm
    SPtemp = np.array([SP1err,SP2err,SP3err])

    return Aout, q, v, w, Alocnorm, SPtemp