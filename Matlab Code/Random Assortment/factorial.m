function [n] = factorial(n)
    if n == 0
        n = 1;
        return;
    else
        n = n * factorial(n-1);
    end