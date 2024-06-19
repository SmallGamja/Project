% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman

% Script to systematically run multiple optimization problems with various methods and step types

% Close all figures, clear all variables from workspace and clear command window



% When running this code, please comment the line in optsolver that
% includes opt_val we
close all;
clear all;
clc;

addpath(genpath('Methods'));
addpath(genpath('Functions'));

options.term_tol = 1e-6;
options.max_iterations = 1e3;

% Define the problemsproblems = struct();

problems(1).name = 'P1_quad_10_10';
problems(1).x0 = 20 * rand(10, 1) - 10;

problems(2).name = 'P2_quad_10_1000';
problems(2).x0 = 20 * rand(10, 1) - 10;

problems(3).name = 'P3_quad_1000_10';
problems(3).x0 = 20 * rand(1000, 1) - 10; % Corrected to match problem dimension

problems(4).name = 'P4_quad_1000_1000';
problems(4).x0 = 20 * rand(1000, 1) - 10; % Corrected to match problem dimension

problems(5).name = 'P5_Quartic_1';
problems(5).x0 = [cos(70); sin(70); cos(70); sin(70)];

problems(6).name = 'P6_Quartic_2';
problems(6).x0 = [cos(70); sin(70); cos(70); sin(70)];

problems(7).name = 'P7_Rosenbrock_2';
problems(7).x0 = [-1.2; 1];

problems(8).name = 'P8_Rosenbrock_100';
problems(8).x0 = [-1.2; ones(99, 1)]; % Corrected to match problem dimension

problems(9).name = 'P9_Datafit_2';
problems(9).x0 = [1; 1];

problems(10).name = 'P10_Exponential_10';
problems(10).x0 = [1; zeros(9, 1)];

problems(11).name = 'P11_Exponential_1000';
problems(11).x0 = [1; zeros(99, 1)]; % Corrected to match problem dimension

problems(12).name = 'P12_Genhumps_5';
problems(12).x0 = repmat(-506.2, 5, 1);

% Set the dimension n for each problem
for i = 1:length(problems)
    problems(i).n = length(problems(i).x0);
end
% Define methods and their options
methods = struct();
methodTypes = {'GradientDescent', 'Newton', 'BFGS', 'TRNewtonCG', 'TRSR1CG', 'DFP'};
stepTypes = {'Backtracking', 'Wolfe'}; % 'Wolfe' not applicable for TR methods




% Iterate over each problem
for p = 1:length(problems)
    problem = problems(p);

    % Iterate over each method
    for m = 1:length(methodTypes)
        for s = 1:length(stepTypes)
            if ~startsWith(methodTypes{m}, 'TR') || strcmp(stepTypes{s}, 'Backtracking') % Skip Wolfe for TR methods
                method.name = methodTypes{m};
                method.options.step_type = stepTypes{s};
                method.options.a_bar = 1;
                method.options.tau = 0.5;
                method.options.c1_ls = 1e-4;
                method.options.beta = 1e-6;
                method.options.eps = 1e-6;
                method.options.c = 0.5;
                method.options.delta = 10;
                method.options.c1_tr = 1e-3;
                method.options.c2_tr = 0.5;
                method.options.term_tol = 1e-6;
                method.options.max_iterations = 1e3;
                method.options.max_iterations_CG = 1e3;
                method.options.term_tol_CG = 1e-6;
                if strcmp(stepTypes{s}, 'Wolfe')
                    method.options.a_bar = 1;
                    method.options.c1_ls = 1e-4;
                    method.options.c2_ls = 0.6;
                    method.options.a_high = 1000;
                    method.options.a_low = 0;
                    method.options.c = 0.5;
                end

                % Run solver
                tic;
                [x, f] = optSolver_Team_Beekman(problem, method, options);
                toc;

                % Display results
                fprintf('Problem: %s, Method: %s, Step Type: %s, Objective Value f: %f\n', problem.name, method.name, method.options.step_type, f);
                fprintf('------------------------------------------\n');
            end
        end
    end
end





