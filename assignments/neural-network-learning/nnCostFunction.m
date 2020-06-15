function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% From instructions:
%   Implementation Note: The matrix X contains the examples in rows (i.e., X(i,:)’
%   is the i-th training example x(i), expressed as a n × 1 vector.) When you
%   complete the code in nnCostFunction.m, you will need to add the column of 1’s
%   to the X matrix.
X = [ones(m, 1) X];

% Transpose, makes following operations a bit nicer.
X = X';

% 1.3 - Foward propagation
%   For context, see Week 4's Forward Propagation: Vectorized Implementation.
a1 = X;

%   Hidden layer
z2 = Theta1 * a1;
a2 = sigmoid(z2);

%   Add bias unit
a2 = [ones(1, m); a2];

%   Output layer
z3 = Theta2 * a2;
a3 = sigmoid(z3);
h_Theta = a3;

% 1.3 - Cost function
%   From instructions:
%      Also, recall that whereas the original labels (in the variable y) were
%      1, 2, ..., 10, for the purpose of training a neural network, we need to
%      recode the labels as vectors containing only values 0 or 1, so that ...
Y = zeros(num_labels, m);
for i = 1:num_labels
	Y(i, :) = (y == i);
end

%   Calculate cost function (without regularization)
J = (1 / m) * sum(sum((-Y) .* log(h_Theta) - (1 - Y) .* log(1 - h_Theta)));


% 1.4 Regularization
%   Note that you should not be regularizing the terms that correspond to the bias.
%   For the matrices Theta1 and Theta2, this corresponds to the first column of
%   each matrix.
Theta1WithoutBias = Theta1(:, 2:size(Theta1, 2));
Theta2WithoutBias = Theta2(:, 2:size(Theta2, 2));
J = J + (0.5 * lambda / m) * (sum(sum(Theta1WithoutBias .^ 2)) + 
	sum(sum(Theta2WithoutBias .^ 2)));


% 2 Backpropagation
%   Implement the backpropagation algorithm to compute the gradient for the neural
%   network cost function.
%   
%   For calculations see Backpropagation Algorithm part of week 5, specifically how
%   it can be written vectorized. First get error term d (partial), then calculate
%   D (big delta).
d3 = (a3 - Y);
D2 = d3 * a2';

d2 = (Theta2WithoutBias' * d3) .* sigmoidGradient(z2);
D1 = d2 * a1';

Theta1_grad = (1 / m) * D1;
Theta2_grad = (1 / m) * D2;

% 2.5 Regularized Neural Networks
%   "Note that you should not be regularizing the first column of Θ(l) which is used
%   for the bias term." - means we reuse the ThetaWithoutBias we've created before.
Accumulator1 = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1WithoutBias];
Accumulator2 = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2WithoutBias];

Theta1_grad += Accumulator1;
Theta2_grad += Accumulator2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
