function [primes] = prime(N)
    primes = 2;
    length_primes = 1;
    for n = 3:2:N-1
        flag = 0;
        for i=1:length_primes
            if mod(n,primes(i)) == 0
                flag = 1;
            end
        end
        if flag == 0
            length_primes = length_primes + 1;
            primes = [primes,n];
        end
    end