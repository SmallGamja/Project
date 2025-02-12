

% Problem Number: 4
% Problem Name: quad_1000_1000
% Problem Description: A randomly generated convex quadratic function; the 
%                      random seed is set so that the results are 
%                      reproducable. Dimension n = 1000; Condition number
%                      kappa = 1000

% function that computes the gradient of the quad_1000_1000 function
function [g] = quad_1000_1000_grad(x, Q, q)
g = Q*x + q;

end