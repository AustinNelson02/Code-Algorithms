function [y] = ex(epsilon,x,N)
    term = 1.0;
    y = term;
    diff= 2*epsilon;
    k = 2;

    while diff > epsilon && k < N
        term = term * x / (k - 1);
        y = y + term;
        diff = abs(term);
        k = k+1;
    end