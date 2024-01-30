function [A] = Cholesky(A)
    [n,n] = size(A);

    for j = 1:n
        A(j,j) = sqrt(A(j,j));
        for i = j+1:n
            A(i,j) = A(i,j)/A(j,j);
        end
        for k = j+1:n
            for i = k:n
                A(i,k) = A(i,k) - A(i,j)*A(k,j);
            end
        end
    end

    for i = 1:n-1
        for j = i+1:n
            A(i,j) = 0;
        end
    end