d = 50;
r = 300;
new_data = 1;
if new_data == 1
	X = eye(d);%normrnd(0, 1, d, r);
	%[Q,R] = qr(X);
	%X = Q(:, 1:r);
	X = bsxfun(@times, 1./sqrt(sum(X.^2)), X);
	%T = fac2tensor_sym(X, ones(r, 1));
end

y_0 = normrnd(0, 1, d, r) * 0.001;
step_size = 0.4;
max_iteration = 10000;
gap = 100;
y = gradient_descent_polynomial(X, y_0, max_iteration, step_size, gap);

%fprintf('init')
%proj_0 = abs(X' * y_0)
%fprintf('final')
%abs(X' * y)
%eigs = svd(y);
