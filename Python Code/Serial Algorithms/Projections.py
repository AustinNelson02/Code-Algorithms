import numpy as np
import copy
from LinearSolvers import *

# Finds the projection of v onto the hyper plane Cz = d
def onto_hyperplane(v,C,d):
    #Find the LS solution of C^Tu = v
    u = LSQR(np.transpose(C),v)

    #Find the project of v onto R(C^T)
    q = np.matmul(np.transpose(C),u)

    #Find the projection of v onto N(C)
    r = v - q

    # Finding the minimum norm solution to Cx0 = d
    x0 = LSQR_eps(C,d)

    x = x0 + r

    return x

def onto_halfspace(v,C,d):
    m = len(C)

    flag = 0
    for i in range(m):

        if np.matmul(C[i,:],v) > d[i]:
            flag = 1

    if flag == 0:
        x = v
    else:
        x = onto_hyperplane(v,C,d)
    return x
