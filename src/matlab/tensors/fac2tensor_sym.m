function T = fac2tensor_sym(factors, eigs)
	[d, rank] = size(factors);
	T = zeros(d, d, d);
	for i = 1:rank
		T = T + eigs(i) * vec2tensor_sym(factors(:,i));
	end
end
