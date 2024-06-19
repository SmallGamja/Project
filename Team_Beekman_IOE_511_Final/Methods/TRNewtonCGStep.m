% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman


function [x_new, f_new, g_new, H_new, delta, func_evals, grad_evals] = TRNewtonCGStep(x, f, g, H, delta, problem, method, options, k)
% Implements a Trust Region Newton-Conjugate Gradient Step for optimization.
%
% Inputs:
%   x - current point.
%   f - current function value at x.
%   g - gradient of the function at x.
%   H - Hessian of the function at x.
%   delta - current trust region radius.
%   problem - structure containing problem-specific functions like compute_f.
%   method - structure containing method-specific options.
%   options - additional options, not used directly in this function.
%   k - current iteration number (not used in the function body).
%
% Outputs:
%   x_new - updated point after the trust region step.
%   f_new - function value at the updated point.
%   g_new - gradient at the updated point.
%   H_new - Hessian at the updated point.
%   delta - updated trust region radius.
%   func_evals - number of function evaluations made within this step.
%   grad_evals - number of gradient evaluations made within this step.

% Extract method options for ease of access and readability
c1_tr = method.options.c1_tr; % threshold for acceptable decrease
c2_tr = method.options.c2_tr; % threshold for very good decrease

% Get the descent direction using the CG Steihaug algorithm
d_k = Conjugate_Gradient(g, H, delta, method.options.term_tol);

% Define the quadratic model for the objective function around the current point
m = @(d) f + g.'*d + 0.5*d.'*H*d;

% Initialize counters for the number of function and gradient evaluations
func_evals = 0;
grad_evals = 0;

% Evaluate the objective function at the proposed new point (x + d_k)
% and increase the function evaluation counter
func_evals = func_evals + 1;
% Compute the ratio (rho) of the actual reduction to the predicted reduction
rho = (f - problem.compute_f(x + d_k))/(f - m(d_k));

% Check if the actual decrease is sufficient relative to the model's prediction
if rho > c1_tr
    % The model's prediction is sufficiently accurate; accept the step
    x_new = x + d_k;

    % Compute the objective function, its gradient, and Hessian at the new point
    f_new = problem.compute_f(x_new);
    func_evals = func_evals + 1; % Update function evaluation count
    g_new = problem.compute_g(x_new);
    grad_evals = grad_evals + 1; % Update gradient evaluation count
    H_new = problem.compute_H(x_new);

    % If the accuracy of the model's prediction is very good, increase the trust region radius
    if rho > c2_tr
        delta = 2*delta;
    end
else
    % The model's prediction is not sufficiently accurate; reject the step
    x_new = x;
    f_new = f;
    g_new = g;
    H_new = H;
    delta = delta/2; % Decrease the trust region radius
end
end
