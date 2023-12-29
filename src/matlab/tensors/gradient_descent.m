% Given a 3-d tensor, use gradient descent with initializaiton x_0
function x_result = gradient_descent(T, true_factor, x_0, max_iteration, step_size, gap)
	[d, r] = size(x_0);
	T_flatten = reshape(T, [d, d*d]);
	Z = norm(T_flatten, 'fro');
	stop_condition = 0.0001;
	x = x_0;

	[d, r_true] = size(true_factor);
	% fprintf('start opt...\n');
	for i = 1:max_iteration
		R = fac2tensor_sym(x, ones(r, 1)) - T;
		R_flatten = reshape(R, [d, d*d]);
		x_reshape = fac_flatten(x);
		x = x - step_size * R_flatten * x_reshape;
		% x = bsxfun(@times, 1./sqrt(sum(x.^2)), x);
		err = norm(R_flatten, 'fro') /Z;
		if err <= stop_condition
			fprintf('%d reached stop condition\n', i);
			x_proj = true_factor' * x
			break
		end

		if mod(i, gap) == 0
%			signal = (x * x_kron' - true_factor * true_factor_kron') * U_true * U_true' * x_kron;
%			noise = x * x_kron' * (eye(d*d) - U_true * U_true') * x_kron;
%			proj_kron = norm((eye(d*d) - U_true * U_true') * x_kron);
			fprintf('Iteration %d, error %.12f\n', i, err);
			%fprintf('%.5f\n', max(correlation));
		end
	end
	x_result = x;
	% fprintf('finish opt...\n')
%	z = T_flatten * kron(x, x) - x;
%	lambda = norm(z) / norm(x)
%	z / norm(z)
%	x / norm(x)
end
