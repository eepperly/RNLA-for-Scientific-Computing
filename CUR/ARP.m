% Implements Adaptive Random Pivoting for CCSP
function J = ARP(V)
    [n, r] = size(V);
    J = zeros(1, r);
    for k = 1:r
        % % Simpler, without Householder reflection
        % p = vecnorm(V');
        % p = p/sum(p);
        % jk = randsample(n,1,true,p);
        % J(k) = jk;
        % x = V(jk,:)';
        % x = x/norm(x);
        % V = V - (V*x)*x';

        % Define the sampling probabilities
        if k<r
            p = vecnorm(V(:,k:r)').^2 / (r-k+1);
        else
            %p = (V(:,r)').^2;
            p = (abs(V(:,r)')).^2;
        end
        p = p.*(p>0);
        if sum(isnan(p)) > 0
            warning("VECTOR p HAS NANs")
            J = J(1:k-1);
            disp(J)
            return;
        elseif max(p) ==0
            warning("VECTOR p IS MADE OF ZEROS")
            J = J(1:k-1);
            return;
        elseif min(p) < 0
            warning("VECTOR p HAS NEGATIVE NUMBERS")
            J = J(1:k-1);
            return;
        end
        p = p/sum(p);


        jk = randsample(n,1,true,p);
        J(k) = jk;
        v = V(jk,k:r)';
        v = v - norm(v)*eye(r-k+1,1);
        if norm(v) == 0
            continue;
        end
        v = v/norm(v);
        V(:,k:r) = V(:,k:r) - 2*V(:,k:r)*v*v';
    end
end