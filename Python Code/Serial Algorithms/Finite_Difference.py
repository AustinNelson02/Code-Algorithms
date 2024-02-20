import numpy as np
import matplotlib.pyplot as plt
import math

## Creating an array of values for the possible parameters h and b
bval = [0,100]

k = np.linspace(0,5,num=1000)
hval = (1/5)*2**(-k)

## Creating the discretization of points between 0 and 1.
#x_i = np.arange(hval[0], 1, hval[0])
#print(x_i)
#b = 2*(hval[1]**2)*x_i
#print(b)
#A = np.diag((-1)*np.ones(len(b)-1),1) + np.diag((2 + bval[1]*hval[0] + hval[0]**2)*np.ones(len(b))) + np.diag(-(1 + bval[1]*hval[0])*np.ones(len(b)-1),-1)
#print(A)

def solution(x,b):
    sol = (((2*b-2-2*b*np.exp(1)**((b-np.sqrt(b**2+4))/2.0))/(np.exp(1)**((b+np.sqrt(b**2+4))/2.0)-np.exp(1)**((b-np.sqrt(b**2+4))/2.0))))*np.exp(1)**(((b+np.sqrt(b**2+4))/2.0)*x)+(2*b-((2*b-2-2*b*np.exp(1)**((b-np.sqrt(b**2+4))/2.0))/(np.exp(1)**((b+np.sqrt(b**2+4))/2.0)-np.exp(1)**((b-np.sqrt(b**2+4))/2.0))))*np.exp(1)**(((b-np.sqrt(b**2+4))/2.0)*x)+2*x-2*b    
    return sol

## Constructing The discretization x_0, A, and b
def construct_A_x_b(h,b_0,option):
    ## Constructing x_0, A, and b
    x_i = np.arange(h,1,h)
    b = 2*(h**2)*x_i
    A = np.diag((-1)*np.ones(len(b)-1),1) + np.diag((2 + b_0*h + h**2)*np.ones(len(b))) + np.diag(-(1 + b_0*h)*np.ones(len(b)-1),-1)
    ## Solving Ax = b
    x = np.linalg.solve(A,b)
    sol = solution(x_i,b_0)

    ## Computing the max norm of the solution vector.
    norm_x = max(abs(x-sol))

    ## If option 0 is given, we only return the max norm
    ## Otherwise, we do return the solution.
    if option == 0:
        return norm_x
    else:
        return x,sol,x_i


sol = construct_A_x_b(hval[-1],0,0)
x_norm = []
for h in hval:
    sol = construct_A_x_b(h,100,0)
    x_norm.append(sol)
print(min(x_norm))
plt.loglog(hval,x_norm)
plt.xlabel("Step Size h")
plt.ylabel("$||x||_\infty$")
plt.title("$||x||_\infty$ vs h for b = 100")
plt.show()

plt.clf()
[x,sol,x_i] = construct_A_x_b(1/20.0,100,1)
plt.plot(x_i,x, 'r')
plt.plot(x_i,sol,'b')
plt.legend(['Numerical Solution','Solution'])
plt.show()
