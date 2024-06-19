% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman

% Function that: (1) computes the GD step; (2) updates the iterate; and, 
%                (3) computes the function and gradient at the new iterate
% 
%           Inputs: x, f, g, problem, method, options
%           Outputs: x_new, f_new, g_new, d, alpha
%
function [x_new,f_new,g_new, func_evals, grad_evals, d,alpha] = GDStep(x,f,g,problem,method,options)



% search direction is -g
d = -g;

func_evals = 0;
grad_evals = 0;

% determine step size
switch method.options.step_type

    case 'Constant'
        alpha = method.options.constant_step_size;
        x_new = x + alpha*d;
        f_new = problem.compute_f(x_new);
        g_new = problem.compute_g(x_new);

    case 'Backtracking'
        %Constants
        a_bar = method.options.a_bar; 
        tau = method.options.tau;
        c1_ls = method.options.c1_ls;
        func_evals = func_evals+1; 
        %Armijo condition 
        while problem.compute_f(x + a_bar*d) > f + c1_ls *a_bar*g'*d
            a_bar = tau*a_bar; 
            func_evals = func_evals+1;
        end
        alpha = a_bar;  
        x_new = x + alpha*d;                        
        f_new = problem.compute_f(x_new);           
        g_new = problem.compute_g(x_new);  
        func_evals = func_evals+1;
        grad_evals = grad_evals+1; 

    % Weak Wolfe Line Search (Canvas/Project/Armijo_Wolfe.pdf)
    case 'Wolfe' 
        % Constants 
        a_low = 0; a_high = 1000; c = 0.5;   
        a_bar = method.options.a_bar; 

        % Line search parameters
        c1_ls = method.options.c1_ls;
        c2_ls = method.options.c2_ls;
        
        % Subroutine parameters 
        % a_high = method.options.a_high;       %(default: 1000)
        % a_low = method.options.a_low;         %(default: 0)
        % c = method.options.c;                 %(default: 0.5)

        while 1
            func_evals = func_evals+1; 
            if (problem.compute_f(x + a_bar*d)) <= (f + c1_ls * a_bar * g' * d)     % First Wolfe condition
                grad_evals = grad_evals+1;
                if (problem.compute_g(x + a_bar*d)'*d >= c2_ls * g' * d)         % Second Wolfe condition
                    alpha = a_bar;
                    break
                end
            end
            func_evals = func_evals+1; 
            if (problem.compute_f(x + a_bar*d)) <= (f + c1_ls * a_bar * g' * d)
                a_low = a_bar;
            else
                a_high = a_bar;
            end
            a_bar = c*a_low + (1-c)*a_high;              % A combination of the low and high thresholds.
            alpha = a_bar; 
        end
        x_new = x + alpha*d;                        % Iterate update
        f_new = problem.compute_f(x_new);           % New function value
        g_new = problem.compute_g(x_new);           % New gradient value
        func_evals = func_evals+1;
        grad_evals = grad_evals+1; 

end

end

