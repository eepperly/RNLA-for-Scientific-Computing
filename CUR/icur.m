function [errors, C, U, T, R, I, J] = icur(A, block_size, threshold, max_it)
    % INPUTS:
    % A, the matrix being approximated
    % block_size, the number of columns rows and cols to update approx
    % threshold, the desired quality of solution in froebinius norm
    % max_it, the maximum number iterations if zero will be set to 
    % floor(min(n_rows, n_cols) / (block_size)).
    % 
    % OUTPUTS:
    % errors, the list of approximated errors for each iteration
    % C, the columns in the approximation
    % U, the Q matrix in the pseudo inverse of cross approximation
    % T, the R matrix in the pseudo inverse of the cross approximation
    % R, the rows in the approximation
    % I, the selected row indices
    % J, the selected column indices
    [rows, cols] = size(A);
    nA = norm(A, 'fro');
    % If user inputs non zero max_it use it otherwise set it to be rows/blocksize
    naive_maxit = floor(min(rows, cols) / (block_size));
    if max_it == 0
        maxit = naive_maxit;
    else
        maxit = max_it;
    end

    errors = zeros(maxit, 1);
    S = randn(ceil(1.1 * block_size), rows) ./ sqrt(ceil(1.1 * block_size));
    % Generate a second matrix to assist in calculating frobenius norm
    S2 = randn(cols, 10) ./ sqrt(10);
    SA = S * A;
    ResL = SA;
    % Preallocate the index arrays
    I = zeros((block_size) * maxit, 1);
    J = zeros(block_size * maxit, 1);
    C = A;
    U = A;
    T = A;
    R = A;
    start_idx_row = 1;
    start_idx_col = 1;
    for i = 1:maxit
        % return the r matrix corresponding to res
        [J, start_idx_col] = idx_LU(ResL, block_size, J, start_idx_col);
        cidx = J(1:i * block_size);
        % note the most recent columns for Right residual
        recent_cols = J((i-1) * block_size + 1:i * block_size);
        if i > 1
            % Compute the right residual, note that all unselected cols are
            % zero
            Ap = C * (T \ (U' * R(:, recent_cols)));
            ResR = A(:, recent_cols) - Ap;
        else
            ResR = A(:, recent_cols);
        end
        % select the rows from the right residual
        [I, start_idx_row] = idx_LU(ResR', block_size, I, start_idx_row);
        C = A(:, cidx);
        % assuming that if a value is non-zero in I it should be a row
        % index
        ridx = I(I>0);
        R = A(ridx, :);
        % Compute the pseudoinverse term
        [U,T] = qr((C(ridx, :)));
        sketch = SA(:, cidx);
        Est = ((sketch / T) * U') * R;
        ResL = SA -  Est;
        errors(i) = norm(ResL * S2, 'fro') / nA;        
        % If the number of iterations is not prespecified than you can exit
        % early
        if errors(i) < threshold
        %    keyboard
            errors = errors(1:i);
            I = I(1:i*block_size);
            J = J(1:i*block_size);
            break
        end

    end

end


function [I, start] = idx_LU(Res, block_size, I, start)
   Res(:, I(I > 0)) = 0;
   [~, ~, P] = lu(Res', "vector");
   end_idx = start + block_size - 1;
   I(start:end_idx) = P(1:block_size)';
   start = end_idx + 1;
end

