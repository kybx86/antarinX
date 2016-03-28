classdef Regularizers
 
 methods (Static = true )
  
  function regularization = ...
    L2(~, weights, ~, lambda, ~, m, ~, networkModel)
   % -- Computes L2 regularization
   %
   % Args:
   %    weights, {1, Layers-1} cell containing the weight matrices
   %    lambda, scalar, regularization hyperparameter
   %    m, scalar, number of samples in batch
   %    networkModel, [layer1_size, layer2_size,...,layern_size], details
   %                structure of network.
   % Returns:
   %    regularization: scalar value of regularized L2
   
   num_layers = length(networkModel);
   
   CONSTANT = ( lambda / (2*m) );
   sigmaSUM = 0;
   
   for i = 1 : num_layers - 1
    w = weights{i};
    sigmaSUM = sigmaSUM + sum( sum( w(:, 2:end).^2, 2) );
   end
   
   regularization = CONSTANT * sigmaSUM;
   
  end
  
  
  function reg_grad = regularize_weights(weights, lambda, m)
   % -- Computes regularization for gradients
   %
   % Args:
   %    weights, {1, Layers-1} cell containing the weight matrices
   %    lambda, scalar, regularization hyperparameter
   %    m, scalar, number of samples in batch
   % Returns:
   %    reg_grad: cell {1, Layers-1} containing regularized weights for each layer
   
   num_weights = length(weights);
   reg_grad = cell(1, num_weights);
   
   for i = 1 : num_weights
    reg_w = ( lambda / m) * ops.omit_replace_bias(weights{i});
    reg_grad{i} = reg_w;
   end
   
  end
  
  
 end
 
end
