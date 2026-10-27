function [times, relres] = gmres_restarted(A, b, m, chk, restart)
%GMRES_RESTARTED  Restarted GMRES(restart).
%   [times,relres] = GMRES_RESTARTED(A,b,m,chk,restart) runs m total
%   iterations in cycles of length `restart`, reporting cumulative time and
%   the true relative residual at each total iteration count in chk.
%
%   Restarting caps the storage and the orthogonalization cost at `restart`
%   vectors, which is why it is used in practice.  The price is that the
%   Krylov space is discarded at each cycle, so the optimality of GMRES holds
%   only within a cycle and convergence can stagnate.

n      = numel(b);
normb  = norm(b);
times  = nan(numel(chk),1);
relres = nan(numel(chk),1);

t    = 0;
x    = zeros(n,1);
r    = b;
done = 0;                                % total iterations completed

while done < m
    cyc = min(restart, m - done);
    t0  = tic;
    beta = norm(r);
    Q = zeros(n, cyc+1);
    H = zeros(cyc+1, cyc);
    Q(:,1) = r / beta;
    t = t + toc(t0);

    for j = 1:cyc
        t0 = tic;
        w = A*Q(:,j);
        for i = 1:j
            H(i,j) = Q(:,i)' * w;
            w = w - H(i,j)*Q(:,i);
        end
        H(j+1,j) = norm(w);
        if H(j+1,j) > eps*normb
            Q(:,j+1) = w / H(j+1,j);
        end
        t = t + toc(t0);

        c = find(chk == done + j, 1);
        if ~isempty(c)
            t0 = tic;
            e1 = zeros(j+1,1);  e1(1) = beta;
            y  = H(1:j+1,1:j) \ e1;
            xc = x + Q(:,1:j)*y;
            times(c)  = t + toc(t0);
            relres(c) = norm(b - A*xc) / normb;
        end
    end

    % Close the cycle: update the iterate and restart from its residual.
    t0 = tic;
    e1 = zeros(cyc+1,1);  e1(1) = beta;
    y  = H(1:cyc+1,1:cyc) \ e1;
    x  = x + Q(:,1:cyc)*y;
    r  = b - A*x;
    t  = t + toc(t0);

    done = done + cyc;
end

end
