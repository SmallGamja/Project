% IOE 511/MATH 562, University of Michigan
% Code written by: Albert S. Berahas

% Script to run code

% close all figures, clear all variables from workspace and clear command
% window
close all
clear all
clc

addpath(genpath('Methods'));
addpath(genpath('Functions'));


% set options
options.term_tol = 1e-6; 
options.max_iterations = 1e3; 
%options.c_1_ls
%options.c_2_ls
%options.c_1_tr
%options.c_2_tr
%options.term_tol_CG
%options.max_iterations_CG



%% Problem7 - Rosenbrock 2, Initial Point: [-1.2, 1], GradientDescent, Backtracking

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;

method.name = 'GradientDescent';
method.options.step_type = 'Backtracking';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;

tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%% Problem7 - Rosenbrock 2, Initial Point: [-1.2, 1], GradientDescentW, Wolfe

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;

method.name = 'GradientDescent';
method.options.step_type = 'Wolfe';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.c2_ls = 0.8;

tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc
%%  Problem7 - Rosenbrock 2, Initial Point: [-1.2, 1], Modified Newton, Backtracking
%set problem (minimal requirement: name of problem)

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'Newton';
method.options.step_type = 'Backtracking';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.beta = 1e-6;


tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc


%%  Problem7 - Rosenbrock 2, Initial Point: [-1.2, 1], Modified Newton, Wolfe
%set problem (minimal requirement: name of problem)

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'Newton';
method.options.step_type = 'Wolfe';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.c2_ls = 0.2;


tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%% Rosenbrok [-1.2, 1], BFGS
%set problem (minimal requirement: name of problem)

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'BFGS';
method.options.step_type = 'Backtracking';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.eps = 1e-6; 

tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%% Rosenbrok [-1.2, 1], BFGS
%set problem (minimal requirement: name of problem)

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'BFGS';
method.options.step_type = 'Wolfe';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.c2_ls = 0.6;
method.options.eps = 1e-6; 
method.options.a_high = 1000;      %(default: 1000)
method.options.a_low = 0;       %(default: 0)
method.options.c = 0.5; 

tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%% TRNewtonCG
%set problem (minimal requirement: name of problem)

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'TRNewtonCG';

method.options.term_tol = 1e-6;
method.options.max_iterations = 1e3;
method.options.c1_tr = 1e-3;
method.options.c2_tr = 0.5;
method.options.term_tol_CG = 1e-6;
method.options.max_iterations_CG = 1e3;
method.options.beta = 1e-6;
method.options.eps = 1e-6;
method.options.delta = 10;

tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc


%% TRSR1CG
%set problem (minimal requirement: name of problem)

problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'TRSR1CG';

method.options.term_tol = 1e-6;
method.options.max_iterations = 1e3;
method.options.c1_tr = 1e-3;
method.options.c2_tr = 0.5;
method.options.term_tol_CG = 1e-6;
method.options.max_iterations_CG = 1e3;
method.options.beta = 1e-6;
method.options.eps = 1e-6;
method.options.delta = 10;

tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%% Rosenbrock DFP
problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)

method.name = 'DFP';
method.options.step_type = 'Backtracking';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.c2_ls = 0.6;
method.options.eps = 1e-6; 
method.options.a_high = 1000;      %(default: 1000)
method.options.a_low = 0;       %(default: 0)
method.options.c = 0.5; 


tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%% Rosenbrok DFPW
problem.name = 'P11_Exponential_1000';
problem.x0 = [1; zeros(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)


method.name = 'DFP';
method.options.step_type = 'Wolfe';
method.options.a_bar = 1; 
method.options.c1_ls = 1e-4;
method.options.c2_ls = 0.6;
method.options.eps = 1e-6; 
method.options.a_high = 1000;      %(default: 1000)
method.options.a_low = 0;       %(default: 0)
method.options.c = 0.5; 


tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc
%% Plotting
figure(1)
legend( 'GD', ...
        'GDW',...
        'Newton', ...
        'NewtonW',...
        'BFGS', ...
        'BFGSW',...
        'TRNewtonCG',...
        'TRSR1CG', ...
        'DFP', ...
        'DFPW');

xlim([1 20]);
title('Problem 11, Exponential 1000')