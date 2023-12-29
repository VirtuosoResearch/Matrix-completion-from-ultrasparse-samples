function fro_norm = tensor_norm(T)
    [d1, d2, d3] = size(T);
    fro_norm = 0.0;
    for i = 1:d1
        fro_norm = fro_norm + norm(T(i, :, :), 'fro')^2;
    end
    fro_norm = sqrt(fro_norm);
end
