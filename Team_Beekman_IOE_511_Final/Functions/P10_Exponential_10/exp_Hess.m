% Exponential Hessian

function [H] = exp_Hess(x)
    z1 = x(1);
    zi = x(2:end);

    H11 = -2*exp(z1)*(exp(z1) - 1)/(exp(z1) + 1)^3 + 0.1*exp(-z1);
    H22 = diag(12*(zi - 1).^2);
    H = diag([H11; zeros(length(x)-1, 1)]) + [zeros(1, length(x)); zeros(length(x)-1, 1), H22];
end