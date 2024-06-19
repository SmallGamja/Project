% IOE 511/MATH 562, University of Michigan
% Code written by: Albert S. Berahas

% Function that runs a chosen algorithm on a chosen problem
%           Inputs: problem, method, options (structs)
%           Outputs: final iterate (x) and final function value (f)
function [x,f, numFuncEvals, numGradEvals, numIterations] = optSolver_Comparing_Algo_Team_Beekman(problem,method,options)

% set problem, method and options
[problem] = setProblem(problem);
[method] = setMethod(method);
[options] = setOptions(options);

% compute initial function/gradient/Hessian
x = problem.x0;
f = problem.compute_f(x);
g = problem.compute_g(x);
H = problem.compute_H(x); 
numFuncEvals = 0;  % Initialize total number of function evaluations
numGradEvals = 0;  % Initialize total number of gradient evaluations
numIterations = 0; % Initialize total number of iterations
norm_g = norm(g,inf);
norm_g_0 = norm_g;
H_init = eye(problem.n);   


% set initial iteration counter
k = 0;

if strcmp(method.name,'TRNewtonCG') || strcmp(method.name,'TRSR1CG')
    delta = method.options.delta;
end


fplot = zeros(1, options.max_iterations);  
a_bar_values_all = zeros(1, options.max_iterations); % Preallocate for efficiency
actual_iterations = 0; % To track the actual number of iterations


while k < options.max_iterations && norm_g >= options.term_tol*max(1,norm_g_0)
    
    % take step according to a chosen method
    switch method.name
        case 'GradientDescent'
            [x_new,f_new,g_new, func_evals, grad_evals, d,alpha] = GDStep(x,f,g,problem,method,options);

        case 'Newton'
            [x_new,f_new,g_new, H_new,func_evals, grad_evals, d, alpha] = NewtonStep(x,f,g,H,problem,method,options);
            H_old = H; H = H_new;
            
        case 'BFGS'
            [x_new,f_new,g_new,H_init_new, func_evals, grad_evals, d,alpha] = BFGSStep(x, f, g, H_init, problem, method, options, k);
            H_init_old = H_init; H_init= H_init_new;
            
        case 'TRNewtonCG'
            [x_new, f_new, g_new, H_new, delta, func_evals, grad_evals] = TRNewtonCGStep(x, f, g, H, delta, problem, method, options);
            % Update Hessian now.
            H_old = H; H = H_new;

        case 'TRSR1CG'
            [x_new, f_new, g_new, H_new, delta, func_evals, grad_evals] = TRSR1CGStep(x, f, g, H, delta, problem, method, options);
            % Update Hessian estimate now.
            H_old = H; H = H_new;

        case 'DFP'  
            [x_new,f_new,g_new,H_init_new,func_evals, grad_evals, d,alpha] = DFPStep(x,f,g,H_init,problem,method,options);
            H_init_old = H_init; H_init = H_init_new;


        otherwise
            error('Method not implemented yet!')
            
    end
    % update old and new function values
    x_old = x; f_old = f; g_old = g; norm_g_old = norm_g;
    x = x_new; f = f_new; g = g_new; norm_g = norm(g,inf);

    numFuncEvals = numFuncEvals + func_evals;
    numGradEvals = numGradEvals + grad_evals;
    numIterations = k + 1; % k starts from 0, hence increment before assigning

    % actual_iterations = k + 1; % Update this counter to keep track of actual iterations
    % a_bar_values_all(k + 1) = alpha; 
    % a_bar_values_all = a_bar_values_all(1:actual_iterations); 



    %fprintf('At time step %k, function value is %f | norm grad is %f\n', k, f, norm_g, alpha);
    fplot(k+1) = f_old;    

    
    % increment iteration counter
    k = k + 1;
end



disp("Num_iterations: " + numIterations)
disp("Num_func_evals: " + numFuncEvals)
disp("Num_grad_eval: " + numGradEvals)


end