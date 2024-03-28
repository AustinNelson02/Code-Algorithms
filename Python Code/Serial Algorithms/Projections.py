from multipledispatch import dispatch
import numpy as np
import copy
from LinearSolvers import *

# Solves the minimum norm problem
def min_norm(C,d):
    [Q,R,rank] = QR(np.transpose(C),.0001)
    w = forwardSub(np.transpose(R),d)
    x = backwardSub(R,w)

    return np.matmul(np.transpose(C),x)

# Finds the projection of v onto the hyper plane Cz = d
@dispatch(np.ndarray,np.ndarray,np.ndarray)
def projection(v,C,d):
    x = min_norm(C,d - np.matmul(C,v))

    return x + v

@dispatch(np.ndarray,np.ndarray,np.ndarray,int)
def projection(v,C,d,s):
    m = len(C)

    flag = 0
    for i in range(m):
        if s == 0:
            if np.matmul(C[i,:],v) > d[i]:
                flag = 1
        else:
            if np.matmul(C[i,:],v) < d[i]:
                flag = 1

    if flag == 0:
        x = v
    else:
        x = projection(v,C,d)
    return x

@dispatch(np.ndarray,float)
def projection(v,r):
    norm = np.linalg.norm(v)
    if norm <= r:
        return v
    else:
        return v*(r/norm)

def dykstra(v,LP,maxk,tol):

    # We first need to know what the size of v is to make the auxiliary vectors.
    n = len(v)

    # We need to know how many vectors to create based on how many convex sets we look at.
    m = len(LP)

    # We can create the initial vector and what to update by creating a matrix of vectors
    # whose rows is the size of v and columns is the size of how many projections we have.
    X = np.zeros((n,m))

    # We start with projecting the original vector, so make a copy of it into an arbitrary vector x_0
    x_0 = copy.deepcopy(v)
    print(x_0)
    diff = np.zeros((n,1))
    diff_norm = 10
    p = np.zeros((n,1))

    k = 0

    # We need to initialize what the elements of X is by first performing the projection on every
    # vector with respect to v.
    # We can just traverse the actual elements of the list for this.
    while diff_norm > tol and k < maxk:
        for i in range(len(LP)):
            print(x_0 + X[:,[i]])
            # Compute the projection.
            p = projection(x_0 + X[:,[i]],*LP[i])

            # Compute the ith vector of X based on the computed projection.
            X[:,[i]] = X[:,[i]] + x_0 - p

            # updating x_0 to be the computed projection p.
            x_0 = copy.deepcopy(p)

            # Finding the difference of the computed projections so we can find an initial error.
            if(i == 0):
                diff = copy.deepcopy(p)
            else:
                diff = diff - p

        diff_norm = np.linalg.norm(diff)
        k = k + 1

        return X[:,[-1]]

            

