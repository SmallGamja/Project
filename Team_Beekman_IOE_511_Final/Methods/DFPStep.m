% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman



% Acknowledgement !!!!!!!!!!!!!
% While implementing this DFP code, our group used ChatGPT to update the
% hessian estimator



function [x_new,f_new,g_new,H_init_new,func_evals, grad_evals, d,alpha] = DFPStep(x,f,g,H_init,problem,method,options)
% Implements one step of the Davidon-Fletcher-Powell (DFP) optimization algorithm.

% determine the search direction and step size for function minimization.

% Inputs:
%   x 
%   f - function val 
%   g - Gradient 
%   H_init - approx to the inverse Hessian matrix.

% Outputs:
%   x_new - Updated point after the DFP step.
%   f_new - Function value at the new point.
%   g_new - Gradient at the new point.
%   H_init_new - Updated approximation to the inverse Hessian matrix.
%   func_evals - Number of function evaluations performed in this step.
%   grad_evals - Number of gradient evaluations performed in this step.
%   d - Search direction used in this step.


% Initialize search direction using the negative of the product of Hessian approximation and gradient
d = -H_init*g;

% Initialize counters for the number of function evaluations and gradient evaluations
func_evals = 0;
grad_evals = 0;

% Determine the step size based on the specified step type in the method options
switch method.options.step_type
    case 'Backtracking'
        % Constants for backtracking line search
        a_bar = method.options.a_bar; % initial step size
        tau = method.options.tau; % reduction factor for step size
        c1_ls = method.options.c1_ls; 

        % Increment function evaluations counter for the upcoming function evaluation
        func_evals = func_evals + 1;

        % Perform backtracking line search to find a suitable step size
        while problem.compute_f(x + a_bar*d) > f + c1_ls * a_bar * g' * d
            a_bar = tau * a_bar; % Reduce step size
        end
        alpha = a_bar; % Final step size after line search
        x_new = x + alpha * d; % Update point
        f_new = problem.compute_f(x_new); % Evaluate function at new point
        g_new = problem.compute_g(x_new); % Evaluate gradient at new point

        % Calculate s and y for the DFP formula
        s = x_new - x;
        y = g_new - g;

        % Increment function and gradient evaluations counters
        func_evals = func_evals + 1;
        grad_evals = grad_evals + 1;

        % Update the Hessian approximation if the curvature condition is satisfied
        %if s' * y =< method.options.eps * norm(s) * norm(y)
            %disp(method.options.eps * norm(s) * norm(y));
            %disp(s');
            %disp(y);
        if s' * y < method.options.eps * norm(s) * norm(y)
            H_init_new = H_init; % No update if curvature condition is not met
        else
            % Apply the DFP formula to update the Hessian approximation
            %H_init_new = H_init + (y'*H_init)/(y'*H_init*y) + (s*s')/(s'*y);
            H_init_new = H_init - (H_init*y)*(y'*H_init)/(y'*H_init*y) + (s*s')/(s'*y);
        end
        
    case 'Wolfe'
        % Initialization for Wolfe conditions line search
        a_bar = method.options.a_bar; % initial step size
        c1_ls = method.options.c1_ls; % coefficient for Armijo condition
        c2_ls = method.options.c2_ls; % coefficient for curvature condition
        a_high = method.options.a_high; % upper bound for step size
        a_low = method.options.a_low; % lower bound for step size
        c = method.options.c; % interpolation factor for updating step size

        % Perform line search with Wolfe conditions
        while true
            func_evals = func_evals + 1; % Increment function evaluations counter
            if problem.compute_f(x + a_bar*d) <= f + c1_ls * a_bar * g' * d % Check Armijo condition
                grad_evals = grad_evals + 1; % Increment gradient evaluations counter
                if problem.compute_g(x + a_bar*d)' * d >= c2_ls * g' * d % Check curvature condition
                    alpha = a_bar; % Step size satisfying Wolfe conditions
                    break;
                end
            end

            % Update bounds for step size based on evaluation of Wolfe conditions
            if problem.compute_f(x + a_bar*d) <= f + c1_ls * a_bar * g' * d
                a_low = a_bar; % Increase lower bound
            else
                a_high = a_bar; % Decrease upper bound
            end
            a_bar = c * a_low + (1-c) * a_high; % Update step size using interpolation
            alpha = a_bar;
        end
        x_new = x + alpha * d; % Update point
        f_new = problem.compute_f(x_new); % Evaluate function at new point
        g_new = problem.compute_g(x_new); % Evaluate gradient at new point

        % curve pair
        s = x_new - x;
        y = g_new - g;

        func_evals = func_evals + 1; % Increment function evaluations counter
        grad_evals = grad_evals + 1; % Increment gradient evaluations counter

        % Update the Hessian approximation if the curvature condition is satisfied
        if s' * y < method.options.eps * norm(s) * norm(y)
            %disp(s');
            %disp(y);
            H_init_new = H_init; % No update if curvature condition is not met
        else
            % Apply the DFP update formula for Hessian approximation
            H_init_new = H_init - (H_init*y)*(y'*H_init)/(y'*H_init*y) + (s*s')/(s'*y);
        end
end
end





