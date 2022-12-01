from mpi4py import MPI
import numpy as np
import math
import sys
import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#Algorithm for the Jacobi Iteration in parallel form.
#The algorithm takes in the comm variable from the MPI package.
#It also takes in the file destination for the mtx file of A and b
#You will need to update this with your own file destinations.
def ParaJacobi(comm,filename1, filename2):
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Inititaling the necessary variables
    local_M = 0
    local_N = 0
    local_x0 = 0
    local_x = 0
    local_b = 0
    global_start = 0
    global_end = 0
    local_numrows = 0
    global_numrows = 0
    #Reading in the information for the matrix A
    #and assigning its elements to M and N
    with open(filename1) as csvfile:
        freader = csv.reader(csvfile, delimiter = ' ')
        count = 0
        for row in freader:
            if count == 0:
                if rank == 0:
                    pass
            elif count == 1:
                #Obtaining the information on the number of rows, columns, and nnz elements
                a = row
                global_numrows = int(a[0])
                global_numcols = int(a[1])
                nnz = int(a[-1])
                ind = np.linspace(0, global_numrows-1, num = global_numrows)
                if rank == 0:
                    local_ind = np.array_split(ind,size)
                else:
                    local_ind = 0

                #Spreading the neccessary information to the processes and establishing
                #Dimensions for the matrices and vectors
                local_ind = comm.scatter(local_ind, root = 0)

                local_numrows = int(len(local_ind))
                local_numcols = global_numcols
                global_start = int(local_ind[0])
                global_end = int(local_ind[-1])

                local_M = np.zeros((local_numrows,1))
                local_N = np.zeros((local_numrows,local_numcols))
                #Creating the initial guess which is set to an array of 0's
                local_x0 = np.zeros((global_numrows,1))
                local_x = np.zeros((local_numrows,1))
                local_b = np.zeros((local_numrows,1))
            else:
                #Puting the read in information into the correct element of M and N
                #If A = L + D + U, M = D and N = - L - U for the Jacobi Method
                a = row
                global_i = int(a[0]) - 1
                global_j = int(a[1]) - 1
                local_i = global_i - global_start
                local_j = global_j
                val = float(a[-1])

                if(global_i <= global_end) and (global_i >= global_start):
                    if global_i == global_j:
                        local_M[local_i] = val
                    else:
                        local_N[local_i][local_j] = -1 * val
            #Updating the counter associated with reading the lines of the file
            count += 1
            
    #Reading in the information for the vector b and splitting it to
    #the array of processes being used
    with open(filename2) as csvfile:
        freader = csv.reader(csvfile, delimiter = ' ')
        count = 0
        counter = 0
        for row in freader:
            if count == 0:
                pass
            elif count == 1:
                pass
            elif count == 2:
                pass
            elif count == 3:
                pass
            else:
                a = row
                val = float(a[0])
                local_i = counter - global_start
                if(counter <= global_end) and (counter >= global_start):
                    local_b[local_i] = val
                    
                counter += 1 
            count = count + 1

    #Setting the maxiterations and tolerance needed in the algorithm
    #If you want to change the number of iterations or tolerance, you must update these two values
    maxiter = 50
    tol = .00000000001
    #Initializing an error that is above the tolerance so that the algorithm will start
    error = 10
    #Initializing a counter k which denotes the number of iterations the algorithm has gone through
    k = 0
    #Creating a vector x that the process will give their local_x information to
    #Process 0 which will take this vector and compute the error as well as updating the local_x0
    x = np.zeros((global_numrows,1))

    #The algorithm for Jacobi iteration. While the iterations haven't hit the max iterations and the
    #error hasn't hit below the tolerance, the algorithm will continue looping
    while k < maxiter and error > tol:
        
        #Computing q = N*x0 + b across the processes
        q = np.matmul(local_N,local_x0) + local_b

        #Solving Mx = q by solving M[i]*x[i] = q[i]
        #This can be done since M is a diagonal matrix
        for i in range(len(local_M)):
            local_x[i] = q[i,0] / local_M[i]

        #After computing the local_x for all of the processes,
        #The 0 process gathers this information and puts it into a single vector
        comm.Gatherv(local_x,x, root = 0)
        
        if rank == 0:
            #The 0 process computes the error between the current local_x0 and the computed x
            error = np.linalg.norm(x - local_x0)
            
        #The local_x0 is updated with the computed x and is broadcasted to all processes
        local_x0 = comm.bcast(x,root = 0)

        #The updated error is also broadcasted to all processes
        error = comm.bcast(error, root = 0)
        
        #The counter for the iterations is updated
        k = k + 1

    #When the method converges, all of the processes have the solution which is in
    #local_x0. So, that is what we return
    return local_x0

    

value = ParaJacobi(comm, 'C:\\Users\\nelso\\Downloads\\stiffnessmatrix.mtx','C:\\Users\\nelso\\Downloads\\forcingmatrix.mtx')
if rank == 0:
    print(value)
#value = ParaJacobi(comm, 'C:\\Users\\nelso\\Downloads\\smallddmatrix.mtx' ,'C:\\Users\\nelso\\OneDrive\\Documents\\rhs.mtx')
#if rank == 0:
    #print(value)
