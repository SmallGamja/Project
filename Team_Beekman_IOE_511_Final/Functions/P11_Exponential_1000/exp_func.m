% Exponential Function

function [f] = exp_func(x)
    z1 = x(1);
    zi = x(2:end);

    f1 = (exp(z1) - 1) / (exp(z1) + 1) + 0.1*exp(-z1);
    f2 = sum((zi - 1).^4);
    f = f1 + f2;
end