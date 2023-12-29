function y = kron_mat(x)
	[d,r] = size(x);
	y = zeros(d*d, r);
	for j = 1:r
		y(:, j) = kron(x(:, j), x(:, j));
	end
end