function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Starting with setting up some context:
    %
    %   1: We have one variable (univariate linear regression).
    %
    %   2: We have one feature: population, and want to predict: profits.
    %
    %   3: We have two parameters: theta_1 and theta_2, which we want to find the values
    %      for which makes the hypothesis fit our training data closest.
    %
    %   4: Gradient descent algorithm: want to find these optimal values by calculating the
    %      partial derivatives of theta_1, theta_2. In batch (simultaneous update).
    %      Other times we do this until converge, but in this case for "num_iters" iterations.
    %
    %   5: General form for partial derivative: 1/m * sum(h_theta(x^(i)) - y^(i)) * x^(i).
    %       - h_theta of x is the hypothesis, meaning h_theta(x^(i)) = theta_0 + theta_1*x^(i).
    %
    %   6: The first column of - X(:,1) - contains 1s ("the constant"), not represented in the
    %      equation above as its just multiplying theta_0 with 1. Not needed for a non-vectorized
    %      solution.
    %
    %   7: The second column of X - X(:,2) - contains training examples with population,
    %      represented by the x^(i) above.
    %
    %   8: y contains training examples with prices

    previous_theta = theta; % Store previous theta to allow simultaneous update.

    % Straightforward (non-vectorized) solution:
    %theta_1_sum = 0;
    %theta_2_sum = 0;
    %for i = 1:m
    %    theta_1_sum += (previous_theta(1) + previous_theta(2) * X(i,2) - y(i)); % Note: X(i,1) is always 1
    %    theta_2_sum += (previous_theta(1) + previous_theta(2) * X(i,2) - y(i)) * X(i,2);
    %end
    %partial_derivative_1 = (1/m) * theta_1_sum;
    %partial_derivative_2 = (1/m) * theta_2_sum;
    %theta(1) = previous_theta(1) - alpha * partial_derivative_1;
    %theta(2) = previous_theta(2) - alpha * partial_derivative_2;

    % Now, first make sure theta_1 and theta_2 are vectors
    %theta_1 = [];
    %theta_2 = [];
    %for i = 1:m
    %    theta_1(i) = (previous_theta(1) + previous_theta(2) * X(i,2) - y(i)); % Note: X(i,1) is always 1
    %    theta_2(i) = (previous_theta(1) + previous_theta(2) * X(i,2) - y(i)) * X(i,2);
    %end
    %partial_derivative_1 = (1/m) * sum(theta_1);
    %partial_derivative_2 = (1/m) * sum(theta_2);
    %theta(1) = previous_theta(1) - alpha * partial_derivative_1;
    %theta(2) = previous_theta(2) - alpha * partial_derivative_2;

    % Then let us start moving towards a vectorized solution:
    %  - Vectorizing h_theta of x:
    %     1. previous_theta(1) + previous_theta(2) * X(i,2) actually has the general form:
    %     2. previous_theta(1) * X(i,1) + previous_theta(2) * X(i,2) --- X(i,1) is always 1
    %     3. previous_theta is a 2x1 matrix, X a 97x2, and y a 97x1. Instead of a for loop we can
    %        calculate this h_theta through X * previous_theta - y and use sum.
    % Replacing the for loop with this gives:

    %theta_1 = [];
    %theta_2 = [];
    %theta_1 = (X * previous_theta - y); % Note: still no need to multiply by X(:,1) as its only 1s
    %theta_2 = (X * previous_theta - y) .* X(:,2); % Note: want element-by-element multiplication: .*
    %partial_derivative_1 = (1/m) * sum(theta_1);
    %partial_derivative_2 = (1/m) * sum(theta_2);
    %theta(1) = previous_theta(1) - alpha * partial_derivative_1;
    %theta(2) = previous_theta(2) - alpha * partial_derivative_2;

    % Cleaning up (also adding X(:,1) element-by-element multiplication even if it is a no-op here)
    %theta(1) = previous_theta(1) - alpha * (1/m) * sum((X * previous_theta - y) .* X(:,1));
    %theta(2) = previous_theta(2) - alpha * (1/m) * sum((X * previous_theta - y) .* X(:,2));

    % Can written as a for loop:
    %for j=1:length(theta)
    %    theta(j) = previous_theta(j) - alpha * (1/m) * sum((X * previous_theta - y) .* X(:,j));
    %end

    % Last step is to get rid of the last for loop and sum:
    %   As talked about earlier, the hypothesis h_theta is calculated through
    %   X * previous_theta - y, which gives a 97x1 matrix. This is used:
    %     - for theta_0 multiplied by X(:,1)
    %     - for theta_1 multiplied by X(:,2)
    %   And the result should fit the dimension of theta, which is a 2x1 matrix.
    %   By multiplying the transposed X', which is a 2x97 matrix, with h_theta we get the result
    %   and dimension we want:
    theta = previous_theta - alpha * (1/m) * (X' * (X * previous_theta - y));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
