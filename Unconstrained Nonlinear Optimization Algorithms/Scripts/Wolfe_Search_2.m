% IOE 511/MATH 562, University of Michigan
% Code written by: Albert S. Berahas

% Script to run multiple optimization problems with various line search parameters

% Close all figures, clear all variables from workspace and clear command window
close all;
clear all;
clc;


addpath(genpath('Methods'));
addpath(genpath('Functions'));

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
a_bar_values = [0.1, 0.5, 1, 10, 100];
c1_ls_values = [1e-10, 5e-9, 1e-8, 1e-4, 1e-3];
c2_ls_values = [0.4, 0.5, 0.6, 0.9, 0.99];


% Set common options
options.term_tol = 1e-6;
options.max_iterations = 1e3;
method.options.beta = 1e-6;
method.options.eps = 1e-6;
method.options.a_high = 1000;
method.options.a_low = 0;
method.options.c = 0.5; 


% Methods
methods = {'GradientDescent', 'Newton', 'BFGS','DFP'};
step_type = 'Wolfe';


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
            for c1 = c1_ls_values
                for c2 = c2_ls_values
                    method.options.a_bar = a;
                    method.options.c1_ls = c1;
                    method.options.c2_ls = c2;
                    % Display configuration
                    fprintf('Running %s on %s with a_bar=%g, c1_ls=%g\n, c2_ls=%g\n', ...
                            method.name, problem.name, a, c1, c2);

                    % Solve the optimization problem (assuming optSolver_Team_Beekman is your solver function)
                    tic
                    [x,f, numFuncEvals, numGradEvals, numIterations] = optSolver_Comparing_Algo_Team_Beekman(problem,method,options);
                    elapsed_time = toc;
                    % Compute the error metric
                    error_metric = abs(f - optimal_values(p));

                    % Score Function
                    F = 0.2 * elapsed_time^2 + 0.2 * numIterations^2 + 0.2 * numFuncEvals^2 ...
                        + 0.2 * numGradEvals^2 + 0.2 * error_metric^2;
                    score = 1 / sqrt(F) * 100;



                    % Collect results
                    
                    % Display results
                    fprintf('Objective Value f: %f\n', f);
                    fprintf('Error Metric: %f\n', error_metric);
                    fprintf('Score: %f\n', score);
                    fprintf('------------------------------------------\n');

                    results = [results; {problem.name, method.name, a,c1,c2, f, error_metric, elapsed_time, numFuncEvals, numGradEvals, numIterations, score}];
                end
            end
        end
    end
end



% Convert results to table and save to CSV
resultsTable = cell2table(results, 'VariableNames', ...
    {'Problem', 'Method', 'a_bar', 'c1_ls', 'c2_ls', 'ObjectiveValue', 'ErrorMetric', 'ElapsedTime', 'NumFuncEvals', 'NumGradEvals', 'NumIterations', 'Score'});
writetable(resultsTable, 'Wolfe_scores.csv');
