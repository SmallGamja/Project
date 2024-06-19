% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman

function [x_new, f_new, g_new, H_new, func_evals, grad_evals, d, alpha] = NewtonStep(x, f, g, H, problem, method, options)
% Implements one iteration of the Newton optimization algorithm.
%
% Newton's method is an optimization algorithm that uses second-order information
% (Hessian matrix) to find the minimum of a function.

% Initialize parameters
beta = method.options.beta; %Initial 1e-6

% Compute diagonal of Hessian approximation
A = ones(1, size(H, 1)) * (H .* eye(size(H, 1))); 

% Initialize function and gradient evaluations counters
func_evals = 0;
grad_evals = 0;


% Modified Newton
% Compute regularization parameter eta to ensure Hessian is positive definite
if min(A) > 0
    eta = 0; % No regularization needed if Hessian diagonal is positive
else 
    eta = -min(A) + beta; %pd
end


%[R,flag,P] = chol(S) additionally returns a permutation matrix P, which is a preordering of sparse matrix S obtained by amd. If flag = 0,
% then S is symmetric positive definite and R is an upper triangular matrix satisfying R'*R = P'*S*P.
p = 1; % Initialize flag for positive definite check
k = 0; % Initialize iteration counter
while p ~= 0
    if k == 0
        eta_k = eta; % Use initial eta value for the first iteration
    else
        eta_k = max([2 * eta_k, beta]); % Increase eta for subsequent iterations
    end

    % Cholesky factorization of regularized Hessian
    %H + eta_k * eye(size(H, 1))
    [L, flag] = chol(H + eta_k * eye(size(H, 1)));
    %[R,flag] = chol(___) also returns the output flag 
    % indicating whether A is symmetric positive definite.
    % You can use any of the input argument combinations in previous syntaxes.
    % When you specify the flag output, chol does not generate an error if the input matrix is not symmetric positive definite.
    %If flag = 0 then the input matrix is symmetric positive definite and the factorization was successful.
    %If flag is not zero, then the input matrix is not symmetric positive definite and flag is an integer indicating the index of the pivot position where the factorization failed.
    p = flag; % Check for positive definiteness
    k = k + 1; % Increment iteration counter
end

% Compute search direction using Cholesky factorization
d = -(L' * L) \ g;

% Determine step size based on the specified step type
switch method.options.step_type
    case 'Backtracking'
        % Backtracking line search parameters
        a_bar = method.options.a_bar; % Initial step size
        tau = method.options.tau; % Reduction factor for step size
        c1_ls = method.options.c1_ls; % Armijo rule constant for sufficient decrease

        % Increment function evaluations for initial function evaluation
        func_evals = func_evals + 1;

        % Perform backtracking line search to find a suitable step size
        while problem.compute_f(x + a_bar * d) > f + c1_ls * a_bar * g' * d
            a_bar = tau * a_bar; % Reduce step size
            func_evals = func_evals + 1; % Increment function evaluations
        end
        alpha = a_bar; % Final step size after line search
        x_new = x + alpha * d; % Compute new candidate point
        f_new = problem.compute_f(x_new); % Compute function value at new point
        g_new = problem.compute_g(x_new); % Compute gradient at new point
        H_new = problem.compute_H(x_new); % Compute Hessian at new point
        func_evals = func_evals + 1; % Increment function evaluations for Hessian computation
        grad_evals = grad_evals + 1; % Increment gradient evaluations for Hessian computation

    case 'Wolfe' 
        % Wolfe conditions line search parameters
        a_bar = method.options.a_bar; % Initial step size
        c1_ls = method.options.c1_ls; % Wolfe condition constant for sufficient decrease
        c2_ls = method.options.c2_ls; % Wolfe condition constant for curvature condition
        a_low = 0; % Lower bound for step size
        a_high = 1000; % Upper bound for step size
        c = 0.5; % Interpolation factor for updating step size

        % Perform line search with Wolfe conditions
        while true
            func_evals = func_evals + 1; % Increment function evaluations
            if (problem.compute_f(x + a_bar * d)) <= (f + c1_ls * a_bar * g' * d) % Check Armijo condition
                grad_evals = grad_evals + 1; % Increment gradient evaluations
                if (problem.compute_g(x + a_bar * d)' * d >= c2_ls * g' * d) % Check curvature condition
                    alpha = a_bar; % Set step size satisfying both Wolfe conditions
                    break;
                end
            end
            func_evals = func_evals + 1; % Increment function evaluations for Wolfe conditions
            if (problem.compute_f(x + a_bar * d)) <= (f + c1_ls * a_bar * g' * d)
                a_low = a_bar; % Update lower bound for step size
            else
                a_high = a_bar; % Update upper bound for step size
            end
            a_bar = c * a_low + (1 - c) * a_high; % Interpolate new step size
            alpha = a_bar; % Set alpha for the current iteration
        end
        x_new = x + alpha * d;
        f_new = problem.compute_f(x_new); 
        g_new = problem.compute_g(x_new); 
        H_new = problem.compute_H(x_new);
        func_evals = func_evals + 1; % Increment function evaluations for Hessian computation
        grad_evals = grad_evals + 1; % Increment gradient evaluations for Hessian computation
end

end
