function T = vec2tensor(v1, v2, v3)
	T = zeros(length(v1), length(v2), length(v3));
	for i = 1:length(v1)
		for j = 1:length(v2)
			for k = 1:length(v3)
				T(i,j,k) = v1(i) * v2(j) * v3(k);
			end
		end
	end
end