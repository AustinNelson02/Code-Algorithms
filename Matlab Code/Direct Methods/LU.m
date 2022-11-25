function [L,U] = LU(A)
    m = size(A,1)
    U = A
    L = eye(m)
    for i = 1:m-1
        for j = i+1:m
            L(j,i) = U(j,i)/U(i,i)
            U(j,i:m) = U(j,i:m) - L(j,i)*U(i,i:m)
        end
    end
end