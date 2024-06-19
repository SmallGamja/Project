function [H] = rosen_100_Hess(x)
    n = numel(x);
    H = zeros(n, n);
    
    H(1:n-1, 1:n-1) = H(1:n-1, 1:n-1) + diag(2 - 400 .* (x(2:n) - x(1:n-1).^2) + 800 .* x(1:n-1).^2);
    H(n, n) = 200;  
    temp_diag = -400 .* x(1:n-1);
    H(sub2ind(size(H), 1:n-1, 2:n)) = temp_diag; 
    H(sub2ind(size(H), 2:n, 1:n-1)) = temp_diag;  
end