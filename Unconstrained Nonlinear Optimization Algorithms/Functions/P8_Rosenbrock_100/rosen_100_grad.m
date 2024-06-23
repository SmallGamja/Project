

% function that computes the gradient of the Rosenbrock_100 function
function [g] = rosen_100_grad(x)

    % Initialize the length and the gradient vector
    n = numel(x);
    g = zeros(n, 1);

    % compute gradient value
    g(1:n-1) = -2 .* (1 - x(1:n-1)) - 400 .* x(1:n-1) .* (x(2:n) - x(1:n-1).^2);
    g(2:n) = g(2:n) + 200 .* (x(2:n) - x(1:n-1).^2);
end