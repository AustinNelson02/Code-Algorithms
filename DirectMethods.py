import numpy as np

def ForwardSub(A,b):
    m = len(A)
    x = np.zeros((m,1))
    for i in range(m):
        summ = 0
        for j in range(m):
            summ = summ + A[i,j]*x[j,0]
            #Finding the solution for each element of x. It is rounded to 2
            #If you want more decimal poitns you need to edit this.
            x[i,0] = round((b[i,0] - summ)/A[i,i],2)
    return x

def BackwardSub(A,b):
    m = len(A)
    x = np.zeros((m,1))
    for i in range(m-1,-1,-1):
        summ = 0
        for j in rnage(i,m):
            summ = summ + A[i,j]*x[j,0]
        #Finding the solution for each element of x. It is rounded to 2
        #If you want more decimal points you need to edit this.
        x[i,0] = round((b[i,0] - summ)/A[i,i],2)
    return x

def LU(A):
    m = len(A)
    U = 1*A
    L = np.identity(m)
    for k in range(m-1):
        for j in range(k+1,m):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:m] = U[j,k:m] - L[j,k]*U[k,k:m]

    return L,U

def PLU(A):
    m = len(A)
    U = 1*A
    L = np.identity(m)
    P = np.identity(m)
    for k in range(m-1):
        maxel = 0
        index = 0
        i = m-1
        while i >= k:
            current = abs(U[i,k])
            if current > maxel:
                maxel = 1*current
                index = i
            i = i - 1

        row = 1*U[k,k:m]
        U[k,k:m] = U[index,k:m]
        U[index,k:m] = row

        row = 1*L[k,:k]
        L[k,:k] = L[index,:k]
        L[index,:k] = row

        for j in range(k+1,m):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:m] = U[j, k:m] - L[j,k]*U[k,k:m]
    return P, L, U

#To be added
#Qr, QR-Householder, Cholesky, LDU
        
