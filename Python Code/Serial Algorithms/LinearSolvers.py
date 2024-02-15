# This file holds various linear system solvers.
import numpy as np
import copy


#Cholesky factorization
def Cholesky(Q):
    A = copy.deepcopy(Q)
    N = len(A)
    for j in range(0,N):
        A[j,j] = np.sqrt(A[j,j])

        for i in range(j+1,N):
            A[i,j] = A[i,j]/A[j,j]

        for k in range(j+1,N):
            for i in range(k,N):
                A[i,k] = A[i,k] - A[i,j]*A[k,j]
    for i in range(0,N-1):
        for j in range(i+1,N):
            A[i,j] = 0

    return A

#QR Factorization with Householder
def QR_Householder(Q):
    A = copy.deepcopy(Q)
    m = len(A)
    n = len(A[0])

    b = np.zeros([n,1])

    for j in range(0,n):
        [v,b[j]] = householder_vector(A[j:m,j])
        v =v[:, np.newaxis]
        A[j:m,j:n] = np.matmul((np.identity(m-j) - b[j]*np.matmul(v,(np.transpose(v)))),A[j:m,j:n])

        if j < m:
            A[j+1:m,j] = v[1:m-j+1,0]
    
    return A,b
#Computing the Householder Vector
def householder_vector(x):

    n = len(x)

    if n > 1:
        s = np.matmul(np.transpose(x[1:n]),x[1:n])
    else:
        s = 0

    v = np.array([1])
    v = np.concatenate((v,x[1:n]),axis = 0)

    if s == 0:
        beta = 0
    else:
        u = pow((pow(x[0],2) + s),.5)
        if x[0] <= 0:
            v[0] = x[0] - u
        else:
            v[0] = -(s / (x[0] + u))

        beta = 2 * pow(v[0],2) / (s + pow(v[0],2))
        v = v/v[0]
    
    return v,beta

def SimpleQR(A):
    m = len(A)
    n = len(A[0])
    R = np.zeros((n,n))
    Q = np.zeros((m,n))

    R[0,0] = np.norm(A[:,0:1])
    Q[:,0:1] = A[:,0:1]/R[0,0]

    for k in range(1,n):
        vector_proj = np.zeros((m,1))
        for i in range(0,k):
            R[i,k] = np.matmul(np.transpose(Q[:,i:i+1]),A[:,k:k+1])
            vector_proj = vector_proj + R[i,k] * Q[:,i:i+1]
            residual = A[:,k:k+1] - vector_proj
        R[k,k] = np.norm(residual)
        Q[:,k:k+1] = residual/R[k,k]

    return Q,R

def LSQR(A,b):
    [Q,R] = simpleQR(A)
    b = np.matmul(np.transpose(Q),b)
    x = backwardsub(R,b)

    return x

def forwardSub(L,b):
    n = len(L)
    m = len(L[0])

    z = np.zeros([n,1])
    z[0] = b[0] / L[0,0]

    for i in range(1,n):
        total = np.matmul(L[i,:i-1],z[:i-1])

        z[i] = (b[i] - total) / L[i,i]

    return z

def backwardSub(U,b):
    n = len(U)
    m = len(U[0])

    z = np.zeros([n,1])

    z[n-1] = b[n-1] / U[n-1,m-1]

    for i in range(n-1,-1,-1):
        total = np.matmul(U[i,i+1:n],z[i+1:n])

        z[i] = (b[i] - total) / U[i,i]
    return z
        




