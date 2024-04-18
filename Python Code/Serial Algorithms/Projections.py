from multipledispatch import dispatch
import numpy as np
import copy
from LinearSolvers import *

# Solves the minimum norm problem
def min_norm(C,d):
    [Q,R,rank] = QR(np.transpose(C),.0000001)
    w = forwardSub(np.transpose(R),d)
    x = backwardSub(R,w)

    return np.matmul(np.transpose(C),x)

# Finds the projection of v onto the hyper plane Cz = d
@dispatch(np.ndarray,np.ndarray,np.ndarray)
def projection(v,C,d):
    x = min_norm(C,d - np.matmul(C,v))

    return x + v

# Finds the projection of v onto the half plane of Cz = d depending on which side you choose.
# Giving a value of 0 will project onto Cz <= d.
# Giving a value of 1 will project onto Cz >= d.
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

# Finds the projection of v onto the ball of radius r centered at the origin.
@dispatch(np.ndarray,float)
def projection(v,r):
    norm = np.linalg.norm(v)
    if norm <= r:
        return v
    else:
        return v*(r/norm)

# Performs the Dykstra Algorithm to find the projection of v onto multiple convex sets.
# The available convex sets are the hyperplane, halfspace/quadrant, and ball of radius r at the origin.
# The parameters of the function are as follows.
#
# v : The vector we would like to project.
# LP: A list of convex sets to project onto. The format accepts a list of lists, where each element is a list
#     of convex sets.
# maxk: Maximum number of iterations.
# tol: Tolerance for the algorithm.
def dykstra(v,LP,maxk,tol):

    # Find the length of all of the vectors.
    n = len(v)

    # Find how many convex sets are given.
    m = len(LP)

    # Initialize a matrix holding the vectors that will be projected on in the algorithm.
    X = np.zeros((n,m))

    # Initialize the auxiliary vector and projection vector. x_0 will hold the final projection in the end.
    # So, we will return that.
    x_0 = copy.deepcopy(v)
    diff = np.zeros((n,1))
    diff_norm = 10
    p = np.zeros((n,1))
    k = 0

    # Performance of the Dykstra Algorithm for a given list of convex sets.
    while diff_norm > tol and k < maxk:
        for i in range(len(LP)):
            # Compute the projection.
            p = projection(x_0 + X[:,[i]],*LP[i])

            # Compute the ith vector of X based on the computed projection.
            X[:,[i]] = X[:,[i]] + x_0 - p

            # Updating x_0 to be the computed projection p.
            x_0 = copy.deepcopy(p)

            # Finding the difference of the computed projections so we can find an initial error.
            if(i == 0):
                diff = copy.deepcopy(p)
            else:
                diff = diff - p

        # Update conditions on the loop to determine if we converged.
        diff_norm = np.linalg.norm(diff)
        k = k + 1

    # Return the final projection of the algorithm.
    return x_0

