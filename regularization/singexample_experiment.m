A = [0 1 0; 0 0 0; 0 0 2];
B = [1 0 0; 0 0 0; 0 0 1];
E = 10^(-8)* [0 0 1; 0 0 1; 1 1 1];
[Q,R] = qr(randn(3));
[Z,R] = qr(randn(3));

eig(A,B)
eig(A+E,B)
eig(Q*A*Z,Q*B*Z)

