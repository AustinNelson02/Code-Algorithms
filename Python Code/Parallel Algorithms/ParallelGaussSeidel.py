from mpi4py import MPI
import numpy as np
import math
import sys
import csv
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#Algorithm for the Gauss-Seidel Iteration in parallel form.
#The algorithm takes in the comm variable from the MPI package.
#It also takes in the file destination for the mtx file of A and b.
#You will need to update this with your own file destinations.
def ParaGaussSeidel(comm, filename1, filename2):
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Initializing the necessary variables
    local_M = 0
    local_N = 0
    local_x0 = 0
    local_x = 0
    local_b = 0
    global_start = 0
    global_end = 0
    local_numrows = 0
    global_numrows = 0
    indexTracker = 0

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

                #Spreading the neccessary information to the processes and establish
                #Dimensions for the matrices and vectors
                local_ind = comm.scatter(local_ind, root = 0)

                local_numrows = int(len(local_ind))
                local_numcols = global_numcols
                global_start = int(local_ind[0])\
                               
                global_end = int(local_ind[-1])

                local_A = np.zeros((local_numrows,local_numcols))
                local_M = np.zeros((local_numrows,local_numcols))
                local_N = np.zeros((local_numrows,local_numcols))
                #Creating the initial guess which is set to an array of 0's
                local_x0 = np.zeros((global_numrows,1))
                local_x = np.zeros((local_numrows,1))
                local_b = np.zeros((local_numrows,1))
                indexTracker = np.zeros((global_numrows,1))

                #Creating an associated array to the rows of A.
                #This associated array has the information of which process holds the
                #current row of the matrix. This is utilized in solving Mx = q
                if global_numrows % size == 0:
                    SizePerIndex = int(global_numrows/size)
                else:
                    SizePerIndex = int(global_numrows / size)+1
                index = 0
                leftover = global_numrows % size
                flag = False
                k = 0
                for i in range(len(indexTracker)):
                    if global_numrows %size != 0:
                        if (i % SizePerIndex == 0 and i != 0):
                            index = index + 1
                            k = k + 1
                        if(k == leftover) and flag == False:
                            SizePerIndex = SizePerIndex - 1
                            flag = True
                        indexTracker[i,0] = index
                    else:
                        if(i % SizePerIndex == 0 and i != 0):
                            index = index + 1
                        indexTracker[i,0] = index
                    
                #Giving this array to all the processes to be used later in solving Mx = q
                indexTracker = comm.bcast(indexTracker,root = 0)
                
                
            else:
                #Puting the read in information into the correct element of M and N.
                #If A = L + D + U, M = L + D and N = - U.
                a = row
                global_i = int(a[0]) - 1
                global_j = int(a[1]) - 1
                local_i = global_i - global_start
                local_j = global_j

                val = float(a[-1])

                if(global_i <= global_end) and (global_i >= global_start):
                    if global_i >= global_j:
                        local_M[local_i][local_j] = val
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

    #Setting the max=iterations and tolerance needed in the algorithm
    #If you want to change the number of iterations or tolerance, you must update these two values.
    maxiter = 50
    tol = .000000000001
    #Initializing an error that is above the tolerance so that the algorithm will start 
    error = 10
    #Initializing a counter k which denotes the number of iterations the algorithm has gone through
    k = 0
    #Creating a vector z that the process will give their local_x information to
    #Process 0 which will take this vector and compute the error as well as updating the local_x0
    x = np.zeros((global_numrows,1))

    #The algorithm for Gauss-Seidell iteration. While the iterations haven't hit the max iterations and the
    #error hasn't hit below the tolerance, the algorithm will continue looping
    while k < maxiter and error > tol:

        #Computing q = N*x0 + b across the processes
        local_q = local_N @ local_x0 + local_b

        #Solving Mx = q by performing forward substitution (since M = L + U)
        #This is done by solving M[i]*x[i] = q[i] and then broadcasting this information
        #to the rest of the processes who will use the information to eliminate that variable
        #from their equation
        for i in range(global_numrows):
            
            #Finding the process that has the current variable index
            if (i <= global_end) and (i >= global_start):
                #Solving M[i]*x[i] = q[i]
                local_i = i - global_start
                local_x[local_i] = local_q[local_i,0] / local_M[local_i,i]
                val = 1*local_x[local_i]
            else:
                val = 0

            #Broadcasting the value of the solved variable to the other processes
            val = comm.bcast(val, root = indexTracker[i,0])

            #Eliminating the solved variable from the equations in the below rows
            for j in range(i+1,global_numrows):
                if (j <= global_end) and (j >= global_start):
                    local_j = j - global_start
                    local_q[local_j,0] = local_q[local_j,0] - val * local_M[local_j,i]

        #Sending the information from the computed local_x to x in process 0
        comm.Gatherv(local_x,x, root = 0)
        if rank == 0:
            print(x)
            #Computing the error of the computed x and the known local_x0
            error = np.linalg.norm(x - local_x0)

        #Updating the local_x0 of all the processes with the computed x
        local_x0 = comm.bcast(x,root = 0)

        #Updating the error with the new one computed by process 0
        error = comm.bcast(error, root = 0)

        #Updating the counter for the number of iterations
        k = k + 1

    #Returning the solution which is in local_x0 in all of the processes
    return local_x0



#value = ParaGaussSeidel(comm, 'C:\\Users\\nelso\\Downloads\\stiffnessmatrix (1).mtx','C:\\Users\\nelso\\Downloads\\forcingmatrix.mtx')
#if rank == 0:
    #print(value)
value = ParaGaussSeidel(comm,'C:\\Users\\nelso\\Downloads\\smallddmatrix.mtx' ,'C:\\Users\\nelso\\OneDrive\\Documents\\rhs.mtx')
if rank == 0:
    print(value)


