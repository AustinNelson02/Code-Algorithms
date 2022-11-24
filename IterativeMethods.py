import numpy as np

def Jacobi(A, x0, b, maxiter, tol):
    m = len(A)
    M = np.zeros((m,m))
    N = np.zeros((m,m))
    x = np.zeros((m,1))

    for i in range(m):
        M[i,i] = A[i,i]
        for j in range(m):
            if i != j:
                N[i,j] = -1 * A[i,j]

    k = 0
    error = 10

    while k < maxiter and error > tol:
        q = N@x0 + b
        for i in range(m):
            x[i,0] = round(q[i,0]/M[i,i],2)

        error = np.linalg.norm(x-x0)
        k = k+1
        x0 = x

    return x

def GaussSeidel(A, x0, b, maxiter, tol):
    m = len(A)
    M = np.zeros((m,m))
    N = np.zeros((m,m))
    x = np.zeros((m,1))

    for i in range(m):
        for j in range(m):
            if j <= i:
                M[i,j] = A[i,j]
            else:
                N[i,j] = -1 * A[i,j]

    k = 0
    error = 10

    while k < maxiter and error > tol:
        q = N@x0 + b
        x = ForwardSub(M,q)
        error = np.linalg.norm(x - x0)
        k = k+1
        x0 = x

    return x

#Forward and Backward Substitution for solving linear systems

def ForwardSub(A,b):
    m = len(A)
    x = np.zeros((m,1))
    for i in range(m):
        
        summ = 0
        for j in range(m):
            summ = summ + A[i,j]*x[j,0]
        #Finding the solution for each element of x. It is rounded to 2
        #If you want more decimal points you need to edit this.
        x[i,0] = round((b[i,0] - summ)/A[i,i],2)
    return x

def BackwardSub(A,b):
    m = len(A)
    x = np.zeros((m,1))
    for i in range(m-1,-1,-1):
        summ = 0
        for j in range(i,m):
            summ = summ + A[i,j]*x[j,0]
        #Finding the solution for each element of x. It is rounded to 2
        #If you want more decimal points you need to edit this.
        x[i,0] = round((b[i,0] - summ)/A[i,i],2)
    return x

#To be added
#Steepest Descent, Conjugate Gradients, GMRES
