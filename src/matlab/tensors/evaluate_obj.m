% evaluate the objective value for the simultaneous diagonalization method
d = 50;
r = 30;
A = normrnd(0, 1, d, r);
A = bsxfun(@times, 1./sqrt(sum(A.^2)), A);

% random projection
q = normrnd(0, 1, d, 1);
q = q / norm(q);
Lambda = diag(q' * A);

mat_diag = A' * A * Lambda * A' * A;
norm_diag = norm(diag(mat_diag))^2
off_diag = norm(mat_diag, 'fro')^2 - norm_diag
