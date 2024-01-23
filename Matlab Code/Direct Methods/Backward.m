function [x] = Backward(U,z)
    [n,m] = size(U);

    x = zeros([n,1]);

    x(n) = z(n)/U(n,n);

    for i = n-1:-1:1
        sum = 0;
        for j = i+1:n
            sum = sum + U(i,j)*x(j);
        end
        x(i) = (z(i) - sum)/U(i,i);
    end