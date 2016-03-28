classdef Objectives
 
 methods (Static = true )
  
  function scores = cross_entropy(~, H, ~, Y)
   % -- Computes unregularized cross-entropy loss
   %
   % Args:
   %    H, hypothesis, (m x n) calculated after feedforward pass
   %    Y, labels, (m x k) target labels in one-hot encoding
   %
   % Returns:
   %    loss, double, value of computed loss for that batch
   
   [m, ~] = size(Y);
   
   scores = ( (1 / m) * sum( sum( ( -Y.*log(H) - (1-Y).*log(1-H) ), 2) ) );
   
  end
  
  function scores = cross_entropy_adjusted(~, H, ~, Y)
   % -- Computes unregularized cross-entropy loss that uses tanh activation as output.
   % --  H_adj scales the hypothesis to H > 0 to avoid log(x<0).
   %
   % Args:
   %    H, hypothesis, (m x n)  matrix calculated after feedforward pass
   %    Y, labels,  (m x k) target labels in one-hot encoding
   %
   % Returns:
   %    scores, double, value of computed loss for that batch
   
   [m, ~] = size(Y);
   
   H_adj = 0.5 * ( H + 1);  % adjusts H to prevent negative log values from tanh()
   scores = ((1 / m) * sum( sum( (-Y.*log(H_adj) - (1-Y).*log(1-H_adj)), 2) ));
   
  end
  
  function scores = negative_log_likelihood(~, H, ~, Y)
   % -- Computes unregularized negative_log_likelihood
   %
   % Args:
   %    H, hypothesis, (m x n)  matrix calculated after feedforward pass
   %    Y, labels,  (m x k) target labels in one-hot encoding
   %
   % Returns:
   %    scores, double, value of computed loss for that batch
   
   [m, n] = size(H);
   
   [~, y_idx] = max(Y, [], 2); % capture column indices from Y_hot
   
   lin_idx = sub2ind([m, n], 1:m, y_idx');
   top_k   = H(lin_idx); % select columns elements specified by linear_index
   
   scores  = - (1 / m) * sum(log(top_k));
   
  end
  
  
  
  function [scores, dscores] = softmax_loss(~, H, ~, Y)
   % -- Softmax loss implements a softmax activation on H that is then used
   % -- to calculate the negative_log_likelihood loss and the derivative
   % -- of the softmax activation.
   %
   % Args:
   %    H, hypothesis, mxn matrix calculated after feedforward pass
   %    Y, labels, mxk matrix target labels matched accordingly
   %
   % Returns:
   %    scores, double, negative_log_likelihood value on softmax activation and y
   %    dscores, (m x k), derivative w.r.t to softmax.
   %
   %  Note: dscores is the derivative w.r.t. to y = softmax(x). However, a general
   %  derivative for loss functions is simply Hypothesis - Targets. From tests,
   %  general derivative yields better results than dy/dx
   
   
   [m, n] = size(H);
   
   [~, y_idx] = max(Y, [], 2); % capture column indices from Y_hot
   
   % --Softmax activation
   H_shift = H - repmat(max(H, [], 2), [1, n]);
   probs   = exp(H_shift) ./ repmat(sum(exp(H_shift), 2), [1, n]);
   
   % --Equivalent method
   %probs = exp(H - repmat(max(H, [], 2), [1, n]));
   %probs = probs ./ repmat(sum(probs, 2), [1, n]);
   
   % --Equivalent method, but numerically unstable for large values of h
   % probs = exp(h)./repmat(sum(exp(h),2), [1, size(h,2)])
   
   lin_idx = sub2ind([m, n], 1:m, y_idx');
   top_k   = probs(lin_idx); % select columns elements specified by linear_index
   
   % -- Negative_log_likelihood
   scores  = - sum(log(top_k)) / m;
   
   % -- Derivative w.r.t to softmax(x)
   dscores = probs;
   dscores(lin_idx) = dscores(lin_idx) - 1;
   dscores = dscores / m;
   
  end
  
  
  
 end
 
 
end
