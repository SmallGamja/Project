% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman

% Our group carefullt examined the performance and conclude that Newton
% Method would be the bese algorithm for Rosenbrock Function.
% Newton with Wolfe line search had the best performance with 
% a_bar = 1, tau = 0.5, c1_ls = 0.3 

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
%options.c_1_ls
%options.c_2_ls
%options.c_1_tr
%options.c_2_tr
%options.term_tol_CG
%options.max_iterations_CG




%%   Rosen Modified Newton, Backtracking
%set problem (minimal requirement: name of problem)

problem.name = 'P8_Rosenbrock_100';
problem.x0 = [-1.2; ones(99, 1)];
problem.n = length(problem.x0);
problem.opt_val = 3.986624;


% set method (minimal requirement: name of method)
method.name = 'Newton';
method.options.step_type = 'Backtracking';
method.options.a_bar = 1;
method.options.tau = 0.5;  
method.options.c1_ls = 0.3;
method.options.beta = 1e-6;

tic
[x,f] = optSolver_Team_Beekman(problem,method,options);
disp(x);
disp(f);
toc


%% Plotting
figure(1)
legend('NWw: (a\_bar, tau, c1\_ls) = (1, 0.5, 0.3)');

xlim([1 100]);
title('Problem 8, Rosenbrock 100')