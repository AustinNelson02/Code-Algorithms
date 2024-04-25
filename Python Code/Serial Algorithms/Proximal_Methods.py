import numpy as np
from Projections import *

# Performs the proximal gradient method given various convex sets as the constraint.
# The function optimized is fit to be of the quadratic form x^TQx + b^Tx.
# The parameters of the function are as follows.
#
# Q: The matrix from the quadratic form x^TQx + b^Tx.
# b: The vector from the quadratic form x^TQx + b^Tx.
# lc: The list of convex sets to project onto in the prox operator.
# t: The time step for the algorithm.
# kmax: The max number of iterations for the algorithm.
# tol: The tolerance for the algorithm.
def proximal_gradient(Q,b,lc,t,kmax,tol):
    #Find the size of the vectors and establish a blank vector that will be our initial guess.
    n = len(Q[0])
    x = np.zeros((n,1))
    k = 1
    diff = 2 * tol

    #Find the current cost of the function before we begin the algorithm.
    cost_current = np.matmul(np.matmul(np.transpose(x),Q),x) + np.matmul(b,x)    
    
    while diff > tol and k < kmax:
        #Compute the next vector in the iteration.
        y  = x - t*(2*np.matmul(Q,x) + np.transpose(b))
        
        #Project our previously computed vector onto the given convex sets.
        if(len(lc) == 1):
            x = projection(y,*lc[0])
        else:
            x = dykstra(y,lc,50,1e-6)

        #Store the old cost of the function.
        cost_old = copy.deepcopy(cost_current)

        #Compute the new cost of the function.
        cost_current = np.matmul(np.matmul(np.transpose(x),Q),x) + np.matmul(b,x)
        
        #Update our iteration counter and the error of the function cost.
        k = k + 1
        diff = abs(cost_old - cost_current)
    return np.around(x,4)

#Function that performs ridge regression on a given cost function.
def ridge_regression(A,b,kmax,p,tol):
    #Find the size of the vectors and establish a blank vector that will be our initial guess.
    n = len(A[0])
    x = np.zeros((n,1))

    k = 1
    L = np.linalg.norm(np.matmul(np.transpose(A),A),2)

    diff = 2 * tol

    #Find the current cost of the function before we begin the algorithm.
    cost_current = 0.5 * pow(np.linalg.norm(np.matmul(A,x) - b,2),2) + p * pow(np.linalg.norm(x,2),2)

    while diff > tol and k < kmax:
        #Compute the next vector in the iteration.
        y = x - (1/L)* np.matmul(np.transpose(A),np.matmul(A,x)-b)

        #Compute the prox operator of y. In this case, it is given by the following formula.
        x = (1 / ((p/L) + 1)) * y

        #Store the old cost of the function.
        cost_old = cost_current

        #Compute the new cost of the function.
        cost_current = 0.5 * pow(np.linalg.norm(np.matmul(A,x) - b,2),2) + p * pow(np.linalg.norm(x,2),2)

        #Update our iteration counter and the error of the function cost.
        k = k + 1
        diff = abs(cost_old - cost_current)

    return np.around(x,4)

#Function that performs proximal gradient on the lasso problem.
def lasso_problem(A,b,kmax,p,tol):

    n = len(A[0])

    x = np.zeros((n,1))

    k = 1

    L = np.linalg.norm(np.matmul(np.transpose(A),A),2)

    diff = 2 * tol

    cost_current = 0.5 * pow(np.linalg.norm(np.matmul(A,x) - b,2),2) + p*(np.linalg.norm(x,1))

    while diff > tol and k < kmax:

        y = x - (1/L)* np.matmul(np.transpose(A),np.matmul(A,x)-b)

        for i in range(n):
            if y[i] <= -p/L:
                x[i] = y[i] + p/L
            elif y[i] >= p/L:
                x[i] = y[i] - p/L
            else:
                x[i] = 0

        cost_old = cost_current

        cost_current = 0.5 * pow(np.linalg.norm(np.matmul(A,x) - b,2),2) + p * (np.linalg.norm(x,1))

        k = k + 1

        diff = abs(cost_old - cost_current)

    return np.around(x,4)

##########
# Attempt at problem 1
###########

I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.double)
b = np.array([[-2,0,0,-3]]).astype(np.double)
L = 0.5
kmax = 2000
tol = 1e-3

C = np.array([[2,1,1,4],[1,1,2,1]])
d = np.array([[7],[6]])

x = proximal_gradient(I,b,[[C,d]],L,kmax,tol)
print(x)

##########
# Attempt at problem 2
###########

##########
# Attempt at problem 3
###########


###########
# Attempt at problem 6
###########
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[1],[1],[1]])
p = 1
kmax = 2000
tol = pow(10,-16)

x = ridge_regression(A,b,kmax,p,tol)
print(x)

###########
# Attempt at problem 7
###########

p = 1
kmax = 2000
tol = pow(10,-16)

x = lasso_problem(A,b,kmax,p,tol)
print(x)
