n = 10;
lambda = 1;
J = diag(lambda*ones(n,1)) + diag(ones(n-1,1),1);

D = diag( 0.25 * rand(n,1) );

condeig(J+D)