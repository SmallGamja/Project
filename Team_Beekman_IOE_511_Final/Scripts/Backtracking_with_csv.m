% IOE 511/MATH 562, University of Michigan
% Code written by: Albert S. Berahas

% Script to run multiple optimization problems with various line search parameters

% Close all figures, clear all variables from workspace and clear command window
close all;
clear all;
clc;


addpath(genpath('Methods'));
addpath(genpath('IOE_511_Final_Functions'));


% Define a structure array 'problems' where each element is a different optimization problem
problems(1).name = 'P1_quad_10_10';
problems(1).x0 = 20 * rand(10, 1) - 10;

problems(2).name = 'P5_Quartic_1';
problems(2).x0 = [cos(70); sin(70); cos(70); sin(70)];

problems(3).name = 'P7_Rosenbrock_2';
problems(3).x0 = [-1.2; 1];

problems(4).name = 'P9_Datafit_2';
problems(4).x0 = [1; 1];

problems(5).name = 'P10_Exponential_10';
problems(5).x0 = [1; zeros(9, 1)];

problems(6).name = 'P12_Genhumps_5';
problems(6).x0 = repmat(-506.2, 5, 1);

% Optimal values for each problem
optimal_values = [-67.9575, 0, 0, 0, 0, 0];

% Line search parameter combinations
a_bar_values = [0.05, 0.1, 1, 10, 100];
tau_values = [0.1, 0.25, 0.5, 0.75, 0.99];
c1_ls_values = [1e-6, 1e-4, 1e-2, 1e-1, 0.3];

% Set common options
options.term_tol = 1e-6;
options.max_iterations = 1e3;
method.options.beta = 1e-6;
method.options.eps = 1e-6;

% Methods
methods = {'GradientDescent', 'Newton', 'BFGS', 'DFP'};
step_type = 'Backtracking';

% Define weights for scoring function


% Initialize results array
results = {};

% Loop over each problem
for p = 1:length(problems)
    problem = problems(p);
    problem.n = length(problem.x0); % Define the dimension of x0 for each problem
    
    % Loop over each method
    for m = 1:length(methods)
        method.name = methods{m};
        method.options.step_type = step_type;

        % Loop over all combinations of tuning parameters
        for a = a_bar_values
            for tau = tau_values
                for c1 = c1_ls_values
                    method.options.a_bar = a;
                    method.options.tau = tau;
                    method.options.c1_ls = c1;

                    
                    % Display configuration
                    fprintf('Running %s on %s with a_bar=%g, tau=%g, c1_ls=%g\n', ...
                            method.name, problem.name, a, tau, c1);
                    % Solve the optimization problem
                    tic
                    [x,f, numFuncEvals, numGradEvals, numIterations] = optSolver_Comparing_Algo_Team_Beekman(problem,method,options);
                    elapsed_time = toc;
                    % Compute the error metric
                    error_metric = abs(f - optimal_values(p));

                    % Score Function
                    F = 0.2 * elapsed_time^2 + 0.2 * numIterations^2 + 0.2 * numFuncEvals^2 ...
                        + 0.2 * numGradEvals^2 + 0.2 * error_metric^2;
                    score = 1 / sqrt(F) * 100;

                    % Display results
                    fprintf('Objective Value f: %f\n', f);
                    fprintf('Error Metric: %f\n', error_metric);
                    fprintf('Score: %f\n', score);
                    fprintf('------------------------------------------\n');
                
                    % Collect results
                    results = [results; {problem.name, method.name, a, tau, c1, f, error_metric, elapsed_time, numFuncEvals, numGradEvals, numIterations, score}];

                end
            end
        end
    end
end

% Convert results to table and save to CSV
resultsTable = cell2table(results, 'VariableNames', ...
    {'Problem', 'Method', 'a_bar', 'tau', 'c1_ls', 'ObjectiveValue', 'ErrorMetric', 'ElapsedTime', 'NumFuncEvals', 'NumGradEvals', 'NumIterations', 'Score'});
writetable(resultsTable, 'Backtracking_scores.csv');
