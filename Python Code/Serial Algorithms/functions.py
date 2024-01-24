def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def e_x(n,x):
    term = 1.0
    y = term
    for k in range(2,n+1):
        term = term * x / (k-1)
        y = y + term
    return y

def ex(epsilon,x,N):
    term = 1.0
    y = term
    diff = 2 * epsilon
    k = 2

    while diff > epsilon and k < N:
        term = term * x / (k-1)
        y = y + term
        diff = abs(term)
        k = k+1
    return y

print(factorial(10))
print(e_x(2,5))
print(ex(.04,5,100))
