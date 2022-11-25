function [P, L, U] = PLU(A)

m = size(A,1);
U = A;
L = eye(m);
P = eye(m);

for k = 1:m-1
    counter = 0;
    maxel = 0;
    for i = k:m
        current = abs(U(i,k));
        if current > maxel
            maxel = current;
            counter = i;
        end
    end

    row = U(k,k:m);
    U(k,k:m) = U(counter,k:m);
    U(counter,k:m) = row;

    row = L(k,1:k-1);
    L(k,1:k-1) = L(counter,1:k-1);
    L(counter,1:k-1) = row;

    row = P(k,:);
    P(k,:) = P(counter,:);
    P(counter,:) = row;

    for j = k+1:m
        L(j,k) = U(j,k)/U(k,k);
        U(j,k:m) = U(j,k:m) - L(j,k)*U(k,k:m);
    end
end
