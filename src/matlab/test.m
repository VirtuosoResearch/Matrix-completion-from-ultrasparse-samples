dim = 10;
rank = 15;
factors = randn(dim, rank);
eigs = ones(rank, 1);
T = fac2tensor_sym(factors, eigs);

% params
max_iterations = 1000;
step_size = 0.002;
gap = 2000;


% one gradient run
% x_init = randn(dim, 1) * 0.1;
% x_found = gradient_descent(T, factors, x_init, max_iterations, step_size, gap);


% iterative gradient runs
factors_found = iterative_gradient_descent(T, factors, max_iterations, step_size, gap);
[d, overparam_rank] = size(factors_found);

% check reconstruction error
T_reconstructed = fac2tensor_sym(factors_found, ones(overparam_rank, 1));
reconstruction_error = tensor_norm(T_reconstructed - T) / tensor_norm(T);
