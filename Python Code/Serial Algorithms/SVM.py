import numpy as np
from Projections import *


# Function that performs the proximal gradient algorithm for training the hard SVM.
# The parameters of the function are the following:
# 
# X: The part of the training set which will be an mxn matrix where m is the
#    number of data points and n is the number of features.
# Y: Vector of predetermined weights for the data points. This vector will hold
#    either 1 or -1.
# kmax: The maximum number of iterations for the algorithm. It is set to 100000
#       if not specified on call.
# tol: The given tolerance for the algorithm. It is set to 1e-5 if not 
#      specified on call.
# a: A given parameter that effects the time step used in the algorithm.
def hard_svm(X,Y,kmax = 100000,tol = 1e-5, a = 0.5):
    
    # We first determine the size of the training set X.
    m,n = X.shape
    
    # We compute the Hadamard product of X and Y from the training set. That is
    # we apply the weights from Y to the data points in X.
    A = np.zeros((m,n))
    for i in range(n):
        A[:,i] = Y * X[:,i]
    
    # Taking the matrix from above, we produce the symmetric form which is used
    # in our quadratic form.
    Q = np.matmul(A,np.transpose(A))
    
    # Finding the Lipschitz constant which will be used in determining the time
    # step of the algorithm.
    L = np.linalg.norm(A,2)**2
    
    # Computing the time step that will be used in the algorithm.
    t = a/L
    
    # Producing our initial guess for the algorithm.
    u = np.random.rand(m,1)
    
    # Setting up our counter and error for the iterations and tolerance.
    k = 1
    diff = 2 * tol

    # Setting up a vector of ones as well as the resulting vector for our convex set.
    # This will be used within the algorithm when we project.
    one = np.ones((1,m))
    d = np.array([[1],[0]])
    

    
    # Proceeding with the proximal gradient algorithm.
    while diff > tol and  k < kmax:
        
        # We first determine the gradient of the quadratic form.
        gradf = np.matmul(Q,u)
        
        # Hold onto the previous iterate of u before we compute the next iterate.
        uold = u
        
        # Perform the proximal gradient method step. This will give us our next
        # iterate.
        y = u - t*gradf
        u = projection(y,np.vstack((one,np.transpose(y))),d)
        
        # Update our counter and error for the algorithm.
        k = k + 1
        diff = np.linalg.norm(u - uold)

    # Producing the vector v which will be hold the weights for our prediction.
    v = np.matmul(np.transpose(A),u)

    # Computing gamma which will be used for our margin for our prediction.
    gamma = np.matmul(np.matmul(np.transpose(u),Q),u)**0.5

    # Determining what slice of our feature matrix X we will use to compute the vector b.
    index = 0
    for i in range(m):
        if u[i] > 0:
            index = i
            break

    # Copmuting the vector b which will be used in our predictions.
    b = Y[i] * (gamma**2) - np.matmul(X[i],v)

    # Returning the vectors v,b, and gamma which will be used in computing our predictions.
    return v,b,gamma
    

def soft_svm(X,Y,kmax = 100000,tol = 1e-5, a = 0.5, C = 50):

    # We first determine the size of the training set X.
    m,n = X.shape

    # We compute the Hadamard product of X and Y from the training set. That is
    # we apply the weights from Y to the data points in X.
    A = np.zeros((m,n))
    for i in range(n):
        A[:,i] = Y * X[:,i]

    # Taking the matrix from above, we produce the symmetric form which is used
    # in our quadratic form.
    Q = np.matmul(A,np.transpose(A))

    # Finding the Lipschitz constant which will be used in determining the time
    # step of the algorithm.
    L = np.linalg.norm(A,2)**2

    # Computing the time step that will be used in the algorithm.
    t = a/L

    # Producing our initial guess for the algorithm.
    u = np.random.rand(m,1)

    # Setting up our counter and error for the iterations and tolerance.
    k = 1
    diff = 2 * tol

    while diff > tol and k < kmax:

        # We first determine the gradient of the quadratic form.
        gradf = np.matmul(Q,u)

        # Hold onto the previous iterate of u before we compute the next iterate.
        uold = u

        # Perform the proximal gradient method step. This will give us our next
        # iterate.
        y = u - t*gradf
        u = dykstra(y,[[np.transpose(y),np.array([[0]])],[np.identity(m),C*np.ones((m,1)),0]],20,1e-6)

        # Update our counter and error for the algorithm.
        k = k + 1
        diff = np.linalg.norm(u-uold)

        
    # Producing the vector v which will be hold the weights for our prediction.
    v = np.matmul(np.transpose(A),u)

    # Computing gamma which will be used for our margin for our prediction.
    gamma = 1/(np.matmul(np.transpose(v),v))**0.5

    # Computing the hadamard product of Y with u which will be used in constructing our vector b.
    z = Y * u
    indexi = 0
    indexj = 0

    for i in range(m):
        if u[i] < C and u[i] > 0:
            indexi = i
        if u[i] > -C and u[i] < 0:
            indexj = i

    # Computing the vector b which will be used in our weights for the predictions.
    b = 0.5 * (-np.matmul((X[indexi,:] + X[indexj,:]),v) + 1)
