classdef NN
 
 methods (Static = true)
  
  
  %move out
  function [fout, cache] = affine_sig_forward(~, x, ~, w, ~, b)
   %  Computes the forward pass for an affine (fully-connected) layer.
   %
   %  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
   %  examples, where each example x[i] has shape (d_1, ..., d_k). We will
   %  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
   %  then transform it to an output vector of dimension M.
   %
   %  Inputs:
   %  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
   %  - w: A numpy array of weights, of shape (D, M)
   %  - b: A numpy array of biases, of shape (M,)
   %
   %  Returns a tuple of:
   %  - out: output, of shape (N, M)
   %  - cache: (x, w, b)
   
   z = x * w';
   y = sigmoid(z);
   
   if( b == true )
    y = ops.add_bias(y);
   end
   
   fout = y;
   cache = {z, y, w, x};
   
  end
  
  % move out
  function [dz, dy] = affine_sig_backward(~, dout, ~, cache2, ~, b)
   %   Computes the backward pass for an affine layer.
   %
   %   Inputs:
   %   - dout: Upstream derivative, of shape (N, M)
   %   - cache: Tuple of:
   %     - x: Input data, of shape (N, d_1, ... d_k)
   %     - w: Weights, of shape (D, M)
   %
   %   Returns a tuple of:
   %   - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
   %   - dw: Gradient with respect to w, of shape (D, M)
   %   - db: Gradient with respect to b, of shape (M,)
   %   """
   
   [~, ~, w2, x1] = cache2{:};
   
   dy = dout*w2;
   dz = (dy).*sigmoidGradient(x1);
   
   if( b == true )
    dz = ops.remove_bias(dz);
   end

  end
  
  
  % move out 
  function [dz, dy] = affine_sig_backward_last(~, hypothesis, ~, target)
   
   dy = hypothesis - target;
   dz = (dy).*sigmoidGradient(hypothesis);
   
  end
  
  % moveD out
  function loss = cross_entropy_loss(~, H, ~, Y)
   %%% Computes unregularized loss for neural network
   % Args:
   %    h, hypothesis, calculated after feedforward pass
   %    y, labels, target labels matched accordingly
   %
   % Returns:
   %    loss, double, value of computed loss for that batch (all samples)
   
   [m, ~] = size(Y);
   
   % Hadamard product '.' of elementwise multiplication
   loss = ( (1/m) * sum( sum( ( -Y.*log(H) - (1-Y).*log(1-H) ), 2) ) );
   
   
  end
  
  % moved out
  function regularization = ...
    L2_regularization(~, weights, ~, lambda, ~, m, ~, networkModel)
   %%% Computes regularization for neural network
   %    nn_layers_weights:
   %           NetworkParams arr, contains layer weight matrices
   %    lambda: regularization parameter
   %    m: size of samples
   %    networkModel: [layer1_size, layer2_size,...,layern_size], details
   %                structure of network.
   % Returns:
   %    regularization: scalar value of regularized constant for cost function
   
   num_layers = length(networkModel); 
   
   CONSTANT = ( lambda / (2*m) );
   
   sigmaSUM = 0;
   
   for i = 1:num_layers - 1
    %  w = nn_layer_weights(i).layer_weights;
    w = weights{i};
    sigmaSUM = sigmaSUM + sum( sum( w(:, 2:end).^2, 2) );
   end
   
   regularization = CONSTANT * sigmaSUM;
   
  end
  
  % moved out
  function reg_grad = regularize_weights(weights, lambda, m)
   
   num_weights = length(weights);
   reg_grad = cell(1, num_weights);
   
   for i = 1: num_weights
    reg_w = ( lambda / m) * ops.omit_replace_bias(weights{i});
    reg_grad{i} = reg_w;
   end
   
  end
  
  
  
  
  
  
 end
 
end