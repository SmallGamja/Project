% Exponential Grad


function [g] = exp_grad(x)
    z1 = x(1);
    zi = x(2:end);

    g1 = 2*exp(z1)/(exp(z1) + 1)^2 - 0.1*exp(-z1);
    g2 = 4*(zi - 1).^3;
    g = [g1; g2];
end