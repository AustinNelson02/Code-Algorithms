function [Q,R] = MGS(A)
    m = size(A,1);
    n = size(A,2);
    Q = zeros([m,n]);
    R = zeros([n,n]);
    V = zeros([m,n]);
    for i = 1:n
        V(:,i) = A(:,i);
    end
    for i = 1:n
        R(i,i) = norm(V(:,i));
        Q(:,i) = V(:,i)/R(i,i);
        for j = i+1:n
            R(i,j) = Q(:,i)'*V(:,j);
            V(:,j) = V(:,j) - R(i,j)*Q(:,i);
        end
    end
end