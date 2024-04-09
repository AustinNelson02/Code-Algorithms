import numpy as np
from Projections import *


def proximal_gradient(f,fp,lc,L,kmax,tol):
    
    n = len(f[0][0])
    x = np.zeros((n,1))

    k = 1

    diff = 2 * tol

    cost_current = np.matmul(np.matmul(np.transpose(x),f[0]),x) + np.matmul(f[1],x)

    while diff > tol and k < kmax:

        y  = x - (1/L)*(2*np.matmul(f[0],x) + np.transpose(f[1]))

        x = dykstra(y,lc,50,.0001)

        cost_old = cost_current

        cost_current = np.matmul(np.matmul(np.transpose(x),f[0]),x) + np.matmul(f[1],x)

        k = k + 1

        diff = abs(cost_old - cost_current)
    
    return np.around(x,4)

def ridge_regression(A,b,kmax,p,tol):

    n = len(A[0])
    
    x = np.zeros((n,1))

    k = 1

    L = np.linalg.norm(np.matmul(np.transpose(A),A),2)

    diff = 2 * tol

    cost_current = 0.5 * pow(np.linalg.norm(np.matmul(A,x) - b,2),2) + p * pow(np.linalg.norm(x,2),2)

    while diff > tol and k < kmax:

        y = x - (1/L)* np.matmul(np.transpose(A),np.matmul(A,x)-b)

        x = (1 / ((p/L) + 1)) * y

        cost_old = cost_current
        
        cost_current = 0.5 * pow(np.linalg.norm(np.matmul(A,x) - b,2),2) + p * pow(np.linalg.norm(x,2),2)

        k = k + 1

        diff = abs(cost_old - cost_current)

    return np.around(x,4)

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
p = .01
kmax = 2000
tol = pow(10,-16)

#x = ridge_regression(A,b,kmax,p,tol)
#print(x)

###########
# Attempt at problem 7
###########

p = 1
kmax = 2000
tol = pow(10,-16)

#x = lasso_problem(A,b,kmax,p,tol)
#print(x)

###########
# Attempt at problem 1
###########

I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
b = np.array([[-2,0,0,-3]])
L = max(np.linalg.eigvals(I))
kmax = 200
p = .01
tol = .0001

C = np.array([[2,1,1,4],[1,1,2,1]])
d = np.array([[7],[6]])

C1 = np.array([[2,1,1,4]])
d1 = np.array([[7]])

C2 = np.array([[1,1,2,1]])
d2 = np.array([[6]])

x = proximal_gradient([I,b],[I,b],[[C,d],[3.0]],L,kmax,tol)
print(x)

