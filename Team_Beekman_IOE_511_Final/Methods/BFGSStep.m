% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman


    % Acknowledgement !!!!!!!!!!!!!!!!!!
    % While writing this BFGS code, Wanping Dong (IOE511 GSI) helped us
    % formulate             
    % I = eye(size(g,1));
    % H_init_new = (I - rho*s*y')*H_init*(I-rho*y*s') + rho*(s*s');
    % and skipping the update. (had problem with initializing the Hessian)
    % and 
    % gave tips for updating the grad evals in BFGS Code during the office
    % hour 
    


function [x_new,f_new,g_new,H_init_new, func_evals, grad_evals, d,alpha] = BFGSStep(x, f, g, H_init, problem, method, options, k)
% Implements one iteration of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm.
% This quasi-Newton method updates the approximation of the Hessian matrix and computes
% the search direction and step size to potentially minimize the function.

% Inputs:
%   x - Current position in the parameter space.
%   f - Current value of the function at x.
%   g - Gradient of the function at x.
%   H_init - Initial Hessian matrix (or its approximation).
%   problem - Struct containing function handles to the objective function and its gradient.
%   method - Struct containing options and settings for the BFGS method.
%   options - Additional options (not used in this function).
%   k - Iteration number (not used in the function body).

% Outputs:
%   x_new - New position in the parameter space after the step.
%   f_new - New function value at x_new.
%   g_new - New gradient at x_new.
%   H_init_new - Updated Hessian approximation.
%   func_evals - Total number of function evaluations performed in this step.
%   grad_evals - Total number of gradient evaluations performed in this step.
%   d - Direction of the step.
%   alpha - Size of the step taken.

% Initialize the search direction using the negative of the product of Hessian approximation and gradient
d = -H_init*g;

% Initialize counters for function and gradient evaluations
func_evals = 0;
grad_evals = 0;

% Determining step size based on the specified line search type in method options
switch method.options.step_type
    case 'Backtracking'   % Using backtracking line search method
        a_bar = method.options.a_bar; % Initial step size
        tau = method.options.tau; % Factor to reduce step size in each iteration
        c1_ls = method.options.c1_ls; % Coefficient for Armijo condition in line search
        func_evals = func_evals + 1; % Counting function evaluations
        while problem.compute_f(x + a_bar*d) > (f + c1_ls*a_bar*g'*d) % Armijo condition check
            a_bar = tau*a_bar; % Reduce step size
            func_evals = func_evals + 1; % Update function evaluations count
        end
        alpha = a_bar;  
        x_new = x + alpha*d; % Update position                        
        f_new = problem.compute_f(x_new); % Compute new function value          
        g_new = problem.compute_g(x_new); % Compute new gradient
        s = x_new - x; % Compute s = x_new - x
        y = g_new - g; % Compute y = g_new - g 
        rho = 1/(s'*y); % Compute scaling factor rho
        func_evals = func_evals + 1; % Update function evaluations count
        grad_evals = grad_evals+1; % Update gradient evaluations count
        if s'*y < method.options.eps*norm(s)*norm(y)
            H_init_new = H_init; % Use initial Hessian if condition is not met
        else
            I = eye(size(g,1)); % Identity matrix of the same size as gradient
            % BFGS formula to update the Hessian approximation
            H_init_new = (I - rho*s*y')*H_init*(I-rho*y*s') + rho*(s*s');
        end

    case 'Wolfe' % Using Wolfe conditions for line search
        % Constants  
        a_bar = method.options.a_bar; % Initial step size
        c1_ls = method.options.c1_ls; % Coefficient for Armijo condition
        c2_ls = method.options.c2_ls; % Coefficient for curvature condition
        a_high = method.options.a_high; % Upper bound for step size (default: 1000)
        a_low = method.options.a_low; % Lower bound for step size (default: 0)
        c = method.options.c; % Interpolation factor to update step size (default: 0.5)

        % Perform line search with Wolfe conditions
        while true
            func_evals = func_evals + 1;  % Count each function evaluation
            if (problem.compute_f(x + a_bar*d)) <= (f + c1_ls * a_bar * g' * d)  
                grad_evals = grad_evals + 1;  % Count gradient evaluations when Wolfe first condition is checked
                if (problem.compute_g(x + a_bar*d)'*d >= c2_ls * g' * d)  % Check curvature condition
                    alpha = a_bar;  % Set alpha if both Wolfe conditions are met
                    break;
                end
            end
            %func_evals = func_evals + 1;  % Count function evaluation for each check
            func_evals = func_evals + 1;
            if (problem.compute_f(x + a_bar * d)) <= (f + c1_ls * a_bar * g' * d)
                a_low = a_bar;  % Increase the lower bound for step size
            else
                a_high = a_bar;  % Decrease the upper bound for step size
            end
            a_bar = c*a_low + (1-c)*a_high;  % Interpolate new step size
            alpha = a_bar;
            %disp(alpha)
        end
        x_new = x + alpha*d;  % Update point using computed step size
        f_new = problem.compute_f(x_new);  % Compute new function value
        g_new = problem.compute_g(x_new);  % Compute new gradient
        s = x_new - x;  
        y = g_new - g;  % curve pair 
        rho = 1 / (s'*y);  % Calculat rho

        func_evals = func_evals + 1;  % Count the function evaluation
        grad_evals = grad_evals + 1;  % Count the gradient evaluation

        % Update Hessian approximation if the curvature condition is satisfied
        if s'*y < method.options.eps*norm(s)*norm(y)
            H_init_new = H_init;  % skip update
        else
            I = eye(size(g,1));
            H_init_new = (I - rho*s*y')*H_init*(I-rho*y*s') + rho*(s*s');
        end
    end


    end

