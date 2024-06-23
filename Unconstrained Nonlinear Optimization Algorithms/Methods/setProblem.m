% IOE 511/MATH 562, University of Michigan
% Code written by: Albert S. Berahas

% Function that specifies the problem. Specifically, a way to compute: 
%    (1) function values; (2) gradients; and, (3) Hessians (if necessary).
%
%           Input: problem (struct), required (problem.name)
%           Output: problem (struct)
%
% Error(s): 
%       (1) if problem name not specified;
%       (2) function handles (for function, gradient, Hessian) not found
%
function [problem] = setProblem(problem)

% check is problem name available
if ~isfield(problem,'name')
    error('Problem name not defined!!!')
end

% set function handles according the the selected problem
switch problem.name
    case'P1_quad_10_10'
        problem.compute_f = @quad_10_10_func;
        problem.compute_g = @quad_10_10_grad;
        problem.compute_H = @quad_10_10_Hess;

    case'P2_quad_10_1000'
        problem.compute_f = @quad_10_1000_func;
        problem.compute_g = @quad_10_1000_grad;
        problem.compute_H = @quad_10_1000_Hess;

    case'P3_quad_1000_10'
        rng(0);
        q = randn(1000,1);
        Q = sprandsym(1000,0.5,0.1,1);
        problem.compute_f = @(x) quad_1000_10_func(x, Q, q);
        problem.compute_g = @(x) quad_1000_10_grad(x, Q, q);
        problem.compute_H = @(x) quad_1000_10_Hess(x, Q, q);

    case'P4_quad_1000_1000'
        rng(0);
        q = randn(1000,1);
        Q = sprandsym(1000,0.5,1e-3,1);
        problem.compute_f = @(x) quad_1000_1000_func(x, Q, q);
        problem.compute_g = @(x) quad_1000_1000_grad(x, Q, q);
        problem.compute_H = @(x) quad_1000_1000_Hess(x, Q, q);

    case 'P5_Quartic_1'
        problem.compute_f = @quartic_1_func;
        problem.compute_g = @quartic_1_grad;
        problem.compute_H = @quartic_1_Hess;

    case 'P6_Quartic_2'
        problem.compute_f = @quartic_2_func;
        problem.compute_g = @quartic_2_grad;
        problem.compute_H = @quartic_2_Hess;

    case 'P7_Rosenbrock_2'
        problem.compute_f = @rosen_func;
        problem.compute_g = @rosen_grad;
        problem.compute_H = @rosen_Hess;

    case 'P8_Rosenbrock_100' 
        problem.compute_f = @rosen_100_func;
        problem.compute_g = @rosen_100_grad;
        problem.compute_H = @rosen_100_Hess;

    case 'P9_Datafit_2'
        problem.compute_f = @datafit_2_func;
        problem.compute_g = @datafit_2_grad;
        problem.compute_H = @datafit_2_Hess;
        
    case 'P10_Exponential_10'
        problem.compute_f = @exp_func;
        problem.compute_g = @exp_grad;
        problem.compute_H = @exp_Hess;
                
    case 'P11_Exponential_1000'
        problem.compute_f = @exp_func;
        problem.compute_g = @exp_grad;
        problem.compute_H = @exp_Hess;
        
    case 'P12_Genhumps_5'
        problem.compute_f = @genhumps_5_func;
        problem.compute_g = @genhumps_5_grad;
        problem.compute_H = @genhumps_5_Hess;

    case 'Quadratic2'
        load('quadratic2.mat');
        A =  problem.A;
        b =  problem.b;
        c =  problem.c;
        problem.compute_f = @quad_func_2;
        problem.compute_g = @quad_grad_2;
        problem.compute_H = @quad_Hess_2;

    case 'Quadratic10'
        load('quadratic10.mat');
        A =  problem.A;
        b =  problem.b;
        c =  problem.c;
        problem.compute_f = @quad_func_10;
        problem.compute_g = @quad_grad_10;
        problem.compute_H = @quad_Hess_10;
    
    case 'Datafit_2'
        problem.compute_f = @datafit_2_func;
        problem.compute_g = @datafit_2_grad;
        problem.compute_H = @datafit_2_Hess;
    
    case 'Exponential'
        problem.compute_f = @exp_func;
        problem.compute_g = @exp_grad;
        problem.compute_H = @exp_Hess;
    
    otherwise
        
        error('Problem not defined!!!')
end