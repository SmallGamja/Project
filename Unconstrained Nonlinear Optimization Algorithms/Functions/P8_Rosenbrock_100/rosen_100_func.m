

function [f] = rosen_100_func(x)
    k = numel(x);
    f = sum((1 - x(1:k-1)).^2 + 100 .* (x(2:k) - x(1:k-1).^2).^2);
end