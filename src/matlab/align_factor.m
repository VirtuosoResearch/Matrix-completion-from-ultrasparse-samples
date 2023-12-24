% return a score between zero and one
function [alignment_score, align_index] = align_factor(factor_matrix, recovered_factor)
    normalized_factor = recovered_factor / norm(recovered_factor);
    column_vec_norms = vecnorm(factor_matrix);
    cos_sim = normalized_factor' * factor_matrix * diag(column_vec_norms.^(-1));
    [alignment_score, align_index] = max(cos_sim);
    fprintf('found a factor at column %d, with alignment score %.2f\n', align_index, alignment_score);
end