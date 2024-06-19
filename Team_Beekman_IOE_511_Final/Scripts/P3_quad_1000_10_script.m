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
addpath(genpath('IOE_511_Final_Functions'));


% set options
options.term_tol = 1e-6;   %epsilon
options.max_iterations = 1e3; 




%% Problem 3 - Quad_1000_10, GradientDescent, Backtracking

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val =  -1.9186e+03;

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

%%  Problem 3 - Quad_1000_10, GradientDescentW, Wolfe

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val =  -1.9186e+03;

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
%%   Problem 3 - Quad_1000_10 Modified Newton, Backtracking
%set problem (minimal requirement: name of problem)

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val =  -1.9186e+03;


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


%%  Problem 3 - Quad_1000_10 Modified Newton, Wolfe
%set problem (minimal requirement: name of problem)

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val =  -1.9186e+03;

% set method (minimal requirement: name of method)
method.name = 'Newton';
method.options.step_type = 'Wolfe';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.c2_ls = 0.2;
method.options.beta = 1e-6;


tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%% Problem 3 - Quad_1000_10 BFGS, Backtracking
%set problem (minimal requirement: name of problem)

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val =  -1.9186e+03;


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

%% Problem 3 - Quad_1000_10 BFGS, Wolfe
%set problem (minimal requirement: name of problem)

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val =  -1.9186e+03;


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
%%  Problem 3 - Quad_1000_10 TRNewtonCG
%set problem (minimal requirement: name of problem)

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val =  -1.9186e+03;


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


%%  Problem 3 - Quad_1000_10 TRSR1CG
%set problem (minimal requirement: name of problem)

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val = -1.9186e+03;


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

%%  Problem 3 - Quad_1000_10 DFP, Backtracking

problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val = -1.9186e+03;


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

%%  Problem 3 - Quad_1000_10 DFPW, Wolfe
problem.name = 'P3_quad_1000_10';
problem.x0 = 20*rand(1000, 1)-10;
problem.n = length(problem.x0);
problem.opt_val = -1.9186e+03;


% set method (minimal requirement: name of method)


method.name = 'DFP';
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

xlim([1 100]);
title('Problem 3, Quadratic 1000 10')