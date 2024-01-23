function [Q,R] = HouseholderQR(A)
    [m,n] = size(A);
    Qtranspose = eye(m);
    R = A;
    for j = 1:n
        x = R(j:m,j);
        [i,k] = size(x);
        e1 = eye(i);
        e1 = e1(:,1)
        if x(1) >= 0
            u = x - norm(x)*e1;
        else
            u = x + norm(x)*e1;
        end

        if (norm(u) < 10^(-10))
            if j < m
                w = [1;zeros(m-j),1];
            else
                w = 1;
            end
        else
            w = u/norm(u);
        end

        if j > 1
            v = [zeros(j-1,1);w];
        else
            v = w
        end
        P = eye(m) - 2*v*v'
        R = P * R;

        Qtranspose = P*Qtranspose;
     end
     Q = Qtranspose';





