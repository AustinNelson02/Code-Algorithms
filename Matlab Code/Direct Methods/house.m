function [v,b] = house(x)
    n = length(x);

    if n > 1
        s = x(2:n)'*x(2:n);
    else
        s = 0;
    end

    v = [1;x(2:n)];

    if s == 0
        b = 0;
    else
        u = sqrt(x(1)^2 + s);
        if x(1) <= 0
            v(1) = x(1) - u;
        else
            v(1) = -s/(x(1) + u);
        end
        b = 2*v(1)^2/(s + v(1)^2);
        v = v/v(1);
    end