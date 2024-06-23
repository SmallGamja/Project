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
options.term_tol = 1e-6;   %epsilon
options.max_iterations = 1e3; 

%% Problem 2 - Quad_10_1000, GradientDescent, Backtracking

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
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


%% Problem 2 - Quad_10_1000, GradientDescentW, Wolfe

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
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
%%  Problem 2 - Quad_10_1000, Modified Newton, Backtracking
%set problem (minimal requirement: name of problem)

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
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


%%  Problem 2 - Quad_10_1000, Modified Newton, Wolfe
%set problem (minimal requirement: name of problem)

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
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

%% Problem 2 - Quad_10_1000, BFGS, Backtracking
%set problem (minimal requirement: name of problem)

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
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

%% Problem 1 - Quad_10_1000, BFGS, Wolfe
%set problem (minimal requirement: name of problem)

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'BFGS';
method.options.step_type = 'Wolfe';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-4;
method.options.c2_ls = 0.9;
method.options.eps = 1e-6; 
method.options.a_high = 1000;      %(default: 1000)
method.options.a_low = 0;       %(default: 0)
method.options.c = 0.5; 


tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%%  Problem 1 - Quad_10_1000, TRNewtonCG
%set problem (minimal requirement: name of problem)

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'TRNewtonCG';

method.options.term_tol = 1e-6;
method.options.max_iterations = 1e3;
method.options.c1_tr = 1e-4;
method.options.c2_tr = 0.9;
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

%%  Problem 1 - Quad_10_1000, TRSR1CG
%set problem (minimal requirement: name of problem)

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)
method.name = 'TRSR1CG';

method.options.term_tol = 1e-6;
method.options.max_iterations = 1e3;
method.options.c1_tr = 1e-4;
method.options.c2_tr = 0.9;
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

%%  Problem 1 - Quad_10_1000, DFP, Backtracking

problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
problem.n = length(problem.x0);
problem.opt_val = 0;


% set method (minimal requirement: name of method)

method.name = 'DFP';
method.options.step_type = 'Backtracking';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 1e-3;
method.options.c2_ls = 0.7;
method.options.eps = 1e-6; 
method.options.a_high = 1000;      %(default: 1000)
method.options.a_low = 0;       %(default: 0)
method.options.c = 0.5; 


tic
[x,f] = optSolver_Kim_HyunJune(problem,method,options);
disp(x);
disp(f);
toc

%%  Problem 1 - Quad_10_1000, DFPW, Wolfe
problem.name = 'P12_Genhumps_5';
problem.x0 = [-506.2 -506.2 -506.2 -506.2 -506.2].';
problem.n = length(problem.x0);
problem.opt_val = 0;


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
        'BFGSW', ...
        'TRNewtonCG',...
        'TRSR1CG', ...
        'DFP', ...
        'DFPW');

xlim([1 400]);
title('Problem 12, Genhumps 5')