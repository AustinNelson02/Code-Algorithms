import numpy as np
from Projections import *

#Function that performs the proximal gradient algorithm on a given cost function
#and constraints.
def proximal_gradient(Q,b,lc,L,kmax,tol):
    #Find the size of the vectors and establish a blank vector that will be our initial guess.
    n = len(Q[0])
    x = np.random.rand(n,1)
    
    k = 1
    diff = 2 * tol

    #Find the current cost of the function before we begin the algorithm.
    cost_current = np.matmul(np.matmul(np.transpose(x),Q),x) + np.matmul(b,x)    
    
    while diff > tol and k < kmax:
        #Compute the next vector in the iteration.
        y  = x - (1/L)*(2*np.matmul(Q,x) + np.transpose(b))
        
        #Project our previously computed vector onto the given convex sets.
        if(len(lc) == 1):
            x = projection(y,*lc[0])
        else:
            x = dykstra(y,lc,50,.000001)

        #Store the old cost of the function.
        cost_old = copy.deepcopy(cost_current)

        #Compute the new cost of the function.
        cost_current = np.matmul(np.matmul(np.transpose(x),Q),x) + np.matmul(b,x)
        print("Cost Old")
        print(cost_old)
        print("Cost New")
        print(cost_current)
        #Update our iteration counter and the error of the function cost.
        k = k + 1
        diff = abs(cost_old - cost_current)
    print(k)
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

###########
# Attempt at problem 1
###########

I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.double)
b = np.array([[-2,0,0,-3]]).astype(np.double)
L = max(np.linalg.eigvals(I))
kmax = 200
p = .01
tol = .0001

C = np.array([[2,1,1,4],[1,1,2,1]])
d = np.array([[7],[6]])

x = proximal_gradient(I,b,[[C,d],[3.0]],L,kmax,tol)
print(x)

