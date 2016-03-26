function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%*****THIS IS A FOR LOOP IMPLEMENTATION****
% for i=1:m
%     %the x matrix must be transformed into a desing matrix by transposing X
%     %for every sample.
%     
%     hypothesis = theta'*X(1,:)';
%     %hypothesis calculates a batch hypothesis for all samples and does not change unless
%     %theta is updated.
%     
%     J = J + 1/(2*m) * ((hypothesis-y(i))^2);
% end


%*****THIS IS ONE  VECTORIZED IMPLEMENTATION (holds for multivariate)****
%%i have proved this result to be true. 

    %differenceMatrix = X*theta-y; 
    %J = (differenceMatrix' * differenceMatrix)/(2*m);

    
%*****THIS IS ANOTHER  VECTORIZED IMPLEMENTATION (holds for multivariate)****    

    hypothesis = X * theta; %this returns a vector of size (mx1), 
    meanErrorSquared = (hypothesis -y).^2;
    J = sum(meanErrorSquared)/(2*m);
    

% =========================================================================

end
