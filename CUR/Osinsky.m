function I = Osinsky(A,V)
%function I = get_columns_osinsky(A,V)
%GET_COLUMNS_OSINSKY Deterministic column subset selection.

    I = [];
    V = V';
    k = size(V,1);

    Atilde = A - A*V'*V;

    for j = 1:k

        %% Selection criterion
        if j < k
            score = ...
                vecnorm(Atilde) ./ ...
                (vecnorm(V(j:k,:)) + eps);
        else
            score = ...
                vecnorm(Atilde) ./ ...
                (abs(V(end,:)) + eps);
        end

        %% Exclude previously selected columns

        score(I) = Inf;

        %% Select next column

        [~,i] = min(score);
        I = [I,i]; %#ok<AGROW>

        %% Householder update

        v = V(j:k,i);
        v = v - norm(v)*eye(k-j+1,1);

        if sum(abs(v)) == 0
            continue
        end

        v = v/norm(v);

        V(j:k,:) = ...
            V(j:k,:) - 2*v*(v'*V(j:k,:));

        %% Residual update

        Atilde = ...
            Atilde ...
            - Atilde(:,i)*(V(j,:)/V(j,i));

        Atilde(:,I) = 0;

    end
end


function [col_arr,C,W,err] = Osinsky_YN(A,V)
% Given A and V (V approximates right singular vectors, V is nxr), find
% indices col_arr s.t. C = A(:,col_arr) is a good column subset for
% low-rank approx. 
% A good choice is V=rowsketch(A), that is, V = randn(r,m)*A. 
% (technically one is supposed to orthonormalize V). 
% Optionally outputs W s.t. CW approx A. 
% 
% Example: n = 1000;
% A = gallery('randsvd',[n],1e15,[],[],[],1);       A = A'*A; 
% r = 400; V = randn(r,n)*A;  [V,~] = qr(V',0); 
% [ix,C,W] = Osinsky(A,V); 
% norm(A-C*W) % should be O(1e-11)
    [m,n] = size(V); if m<n, V = V'; end
    [m,n] = size(A); 
    A_til = A - (A*V)*V'; % this seems to require O(n^2r) operations. May need to do something about it (we only need the column norms). 
    V = V'; 
    Vori = V; 
    n = size(A,2); 
    r = size(V,1); 
    ix = 1:n;
    err = 0; 
    for k = 1:r
        A_vecnorm = vecnorm(A_til(:,k:n)); 
        V_vecnorm = vecnorm(V(k:r,k:n));

        if k<r
        finder = A_vecnorm./V_vecnorm;
        else
        finder = A_vecnorm./abs(V(r,k:n));
        end
        
        [errnow,j] = min(finder);               
        err = err + errnow^2; 
        j = j+k-1; % key change
        ixtmp = ix(k);
        ix(k) = ix(j);         ix(j) = ixtmp;

        %swap columns
        v = A_til(:, k);
        %if k==r, keyboard, end
        A_til(:, k) = A_til(:, j);
        A_til(:, j) = v;

        v = V(:, k);
        V(:, k) = V(:, j);
        V(:, j) = v;
        %next step
        nu = V(k:r,k);
        nu(1) = nu(1) + sign(nu(1))*norm(nu);
        if nu(1)==0, keyboard, end
        nu = nu/norm(nu);
        V(k:r,1:n) = V(k:r,1:n) - 2*nu*(nu'*V(k:r,1:n));
        A_til = A_til - (1/V(k,k))*(A_til(:,k)*V(k,1:n));
%                if sum(isnan(A_til(:)))>0, keyboard, end        
    end
    
    col_arr = ix(1:r);
    C = A(:,col_arr);    
    Vper = Vori(:,ix);
    if nargout >=3 
    W = Vper(1:r,1:r)\Vori; 
    end
    err = sqrt(err); 
end


