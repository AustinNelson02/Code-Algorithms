import numpy as np
from Projections import *
from LinearSolvers import *


def proximal_gradient(f,C,L,kmax,tol):
    #Establish an initial guess to the solution.
    m = len(f)
    x_0 = np.zeros((m,1))

    #Set up the initial iteration of x and the counter k.
    x = x_0
    k = 1

    #Initialize the error difference.
    diff = 2*tol

    #Find the current cost of the function.
    current_cost = f(x,0)                       ## To be edited for f(x,0)

    while diff > tol and k < kmax:
        #Compute the new vector to be projected.
        y = x - (1/L)*f(x,1)                    ## To be edited for f(x,1)

        #Project the previously computed vector onto the convex sets.
        x = dykstra(y,C,kmax,tol)

        old_cost = current_cost
        k = k + 1

        # Compute the new difference 
        diff = abs(old_cost - current_cost)

    return f(x,0)
