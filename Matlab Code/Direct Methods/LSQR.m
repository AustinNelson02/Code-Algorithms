function [x] = LSQR(A,b)
    [m,n] = size(A);

    [H,beta] = householder(A);

    v = zeros([m,1]);

    for j = 1:n
        R(i,i:n) = H(i,i:n);
        v(j) = 1;
        v(j+1:m) = H(j+1:m,j);
        b(j:m) = (eye(m-j+1) - beta(j)*v(j:m)*v(j:m)')*b(j:m);
    end
    x = Backward(R(1:n,1:n),b(1:n));
