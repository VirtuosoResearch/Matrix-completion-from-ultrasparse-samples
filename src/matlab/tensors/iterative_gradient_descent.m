function recovered_factors = iterative_gradient_descent(T, true_factors, max_iterations, step_size, print_gap)
    
    % get input dimension
    [d, r] = size(true_factors);
    
    % copy tensor data
    T_remaining = T(:, :, :);
    initial_norm = tensor_norm(T_remaining);
    remaining_norm = initial_norm;
    fprintf('at rank %d, with tensor norm %.2f\n', 0, initial_norm);
    
    % init params
    init_scale = 0.05;
    
    current_rank = 0;
    max_search_steps = r * 20;
    recovered_factors = zeros(d, max_search_steps);

    for i = 1:max_search_steps
        x_init = randn(d, 1) * init_scale;
        found_factor = gradient_descent(T_remaining, true_factors, x_init, max_iterations, step_size, print_gap);
        [current_score, current_index] = align_factor(true_factors, found_factor);
        if current_score >= 0.95
            T_remaining = T_remaining - vec2tensor_sym(found_factor); %true_factors(:, current_index)
            current_rank = current_rank + 1;
            recovered_factors(:, current_rank) = found_factor;
            remaining_norm = tensor_norm(T_remaining);
            fprintf('step %d: at rank %d, with tensor norm %.2f\n', i, current_rank, remaining_norm);
        else
            fprintf('step %d: factor not found in the current step\n', i);
        end

        if remaining_norm / initial_norm <= 0.01
            fprintf('succeed!! %d', current_rank);
            recovered_factors = recovered_factors(:, 1:current_rank);
            return
        end
    end
end