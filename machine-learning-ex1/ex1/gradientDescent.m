function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
s = size(theta,1);

%partialTheta0=0;
%partialTheta1=0;

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    hypothesis = X*theta;
    
    %theta_tmp = zeros(s,1);
   
%     for row = 1:s
%         theta_tmp(row) = theta(row)- (1/m)*alpha*sum((hypothesis-y).*X(:,row));        
%     end
%     theta = theta_tmp;
    
    theta = theta - (alpha/m)* X'*(hypothesis-y);
    
    %
    %     partialTheta0 = (partialTheta0 +(theta'*X(1,:)-y)) .*X(iter,1))/m;
    %     partialTheta1 = (partialTheta1 +(theta'*X(1,:)-y)) .*X(iter,2))/m;
    %
    %     temp0= temp0 -alpha*partialTheta0;
    %     temp1= temp1 - alphta*partialTheta1;
    %
    %     theta(1,1)=temp0;
    %     theta(2,1)= temp1; 
    % ============================================================
    
    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
    
end


    
%     figure
%     subplot(2,1,2)
%     plot(J_history)

end
