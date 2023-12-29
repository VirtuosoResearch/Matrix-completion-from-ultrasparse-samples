function X = fac_flatten(factors)
	[d, r] = size(factors);
	X = zeros(d*d, r);
	for i = 1:r
		X(:,i) = reshape(factors(:,i)*factors(:,i)', [d*d,1]);
	end
end
