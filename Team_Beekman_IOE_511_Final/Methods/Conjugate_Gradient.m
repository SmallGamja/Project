% IOE 511/MATH 562, University of Michigan
% Code written by: Team Beakman


% Acknowledgement !!!!!!!!!!!!!!!!!!
% Our group used ChatGPT to calculate the tau updates


function [descent_direction] = Conjugate_Gradient(gradient_at_k, approx_hessian_at_k, trust_region_radius, termination_tolerance)
    % Conjugate_Gradient performs the conjugate gradient method 
    % on a quadratic function within a trust region.

    % Initialize variables:
    % z is the current solution estimate within the trust region
    % r is the residual (negative gradient)
    % p is the conjugate direction
    z = zeros(size(gradient_at_k));
    r = gradient_at_k;
    p = -r;

    % Check if the termination tolerance has already been met
    if norm(r) < termination_tolerance
        % If so, the descent direction is zero (no further optimization needed)
        descent_direction = z;
        return;
    end

    % Start the iterative process
    while true
        % If the curvature condition is not positive, exit the loop
        if p.'*approx_hessian_at_k*p <= 0
            % Solve the trust region subproblem for the scalar tau
            tau = calculateTau(p, z, trust_region_radius);
            descent_direction = z + tau * p;
            return;
        end
        
        % Compute the step length alpha
        alpha = (r.'*r) / (p.'*approx_hessian_at_k*p);
        % Update the solution estimate
        z_new = z + alpha * p;
        
        % If the updated solution estimate exceeds the trust region, solve
        % the subproblem for tau to adjust the descent direction
        if norm(z_new) >= trust_region_radius
            tau = calculateTau(p, z, trust_region_radius);
            descent_direction = z + tau * p;
            return;
        end
        
        % Compute the new residual
        r_new = r + alpha*approx_hessian_at_k*p;
        % Check the termination condition on the norm of the residual
        if norm(r_new) <= termination_tolerance
            descent_direction = z_new;
            return;
        end
        
        % Compute beta for the new conjugate direction
        beta = (r_new.'*r_new) / (r.'*r);
        % Update the conjugate direction
        p = -r_new + beta * p;    
        
        % Update the solution estimate and residual
        z = z_new;
        r = r_new;
    end
end

function tau = calculateTau(p, z, delta_k)
    % calculateTau solves for the scalar tau that is used to ensure the
    % descent direction remains within the trust region boundary.
    
    % Calculate the necessary products for the quadratic formula
    prod1 = p.' * z;
    prod2 = p.' * p;
    prod3 = z.' * z;
    
    % Use the quadratic formula to solve for tau
    % The equation is derived from the trust region constraint ||z + tau*p||^2 = delta_k^2
    tau = (sqrt(prod1^2 + prod2*(delta_k^2 - prod3)) - prod1) / prod2;
end
