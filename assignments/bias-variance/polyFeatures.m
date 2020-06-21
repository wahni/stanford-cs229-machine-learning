function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% 3 Polynomial regression
%   Expanding on the documentation, the X_poly should look as follows:
%     X_poly(1, :) = [X(1) X(1).^2 X(1).^3 ...  X(1).^p];
%     X_poly(2, :) = [X(2) X(2).^2 X(2).^3 ...  X(2).^p];
% 
%   Meaning if we iterate over 1:p, we should iteratve over columns (as
%   opposed to the doc which hints at iterating i over rows).
for i = 1:p
	X_poly(:, i) = X .^ i;
end

% =========================================================================

end
