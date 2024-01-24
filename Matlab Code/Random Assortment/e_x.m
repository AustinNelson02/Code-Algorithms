function [y] = e_x(n,x)
    term = 1.0;
    y = term;
    for k = 2:n
        term = term * x / (k - 1);
        y = y + term;
    end