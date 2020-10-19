function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% assuming X (m x n+1), y (m x 1), theta (n+1 x 1)
% in this case number of features n = 1

% see ex1 computeCost

hThetaX = zeros(size (y));
regTerm = 0;

% linear regression hypothesis
hThetaX = X * theta; % (m x n+1) * (n+1 x 1) result (m x 1)

% regularization term calculation
theta(1) = 0; % theta_0 is not regularized; and theta is a local variable...
% and we have already computed hThetaX

% regTerm = (lambda/2m) * SumOveri_1_to_m(theta_squared)
regTerm = (1/2) * (1/m) * lambda * sum(theta.^2);

% use sum() when doing element operations for sum_over process (i.e. SigmaSum)
% else use vector multiplication to handle sum_over process i.e.
% sum(theta.^2) => (theta' * theta)

% regularized cost
% J_theta = ((1/2) * (1/m) * sum_square_error(hThetaX,y)) + regTerm
J = ((1/2) * (1/m) * sum((hThetaX-y).^2)) + regTerm;

% =========================================================================

% regularized gradient calculation

regTermGrad = 0;

% note gradient => partial derivative
% gradient = 1/m * (SumOveri_1_to_m[hThetaX-y] * X) + regTermGrad

% where 
% regTermGrad = 1/m * lambda * theta_j for J > 0, 
% regTermGrad = 0 for j = 0

% [hthetaX-y] * X => (m x 1)*(m x n+1) => do X' * [hthetaX-y]
% so we get (n+1 x m)*(m x 1) = (n+1 x 1) which is the same as theta

% regTermGrad = scalar value * theta i.e. scalar * (n+1 x 1) i.e. (n+1 x 1)

% hence gradient = (n+1 x 1) + (n+1 x 1) = (n+1 x 1) 

% again SumOveri_1_to_m (i.e. SigmaSum) is handled by vector 
% multiplication inherently

regTermGrad = (1/m) * lambda * theta;
% note - theta(1) = 0 already since theta_0 is not regularized

grad = (1/m) * (X' * (hThetaX - y)) + regTermGrad;

% unrolling grad and returning in case grad is a matrix
grad = grad(:);

end
