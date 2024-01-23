function [A,b] = householder(A)
    [m,n] = size(A);
    b = zeros([n,1]);

    for j = 1:n
        [v,b(j)] = house(A(j:m,j));
        A(j:m,j:n) = (eye(m-j+1) - b(j)*v*v')*A(j:m,j:n);

        if j < m
            A(j+1:m,j) = v(2:m-j+1);
        end
    end