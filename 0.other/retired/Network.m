classdef Network %%% RETIRED
 properties
  
 end
 
 methods (Static = false)
  
  function obj = Network()
   %empty constructor 
  end
  
  
  
  
 end
 
 methods (Static = true)
  
  
  function Network_weights_arr = nn_weights(net_layers)
   %%% Creates layer weights for network of any size
   % Args:
   %    array: net_layers, [size_layer1, size_layer2 , ... ,size_layerN]
   %    NOTE: size of layers must be UNBIASED
   % Returns:
   %    NetworkParams array: array of weights for all layers in nn
   
   
   num_layers =  length(net_layers);
   
   % Pre-allocation of Network type array
   Network_weights_arr = repmat(NetworkParams(), 1, num_layers);
   
   for a = 1:num_layers - 1 %there will be one weight less than layers
    
    L_in   = net_layers(a);
    L_out  = net_layers(a+1);
    
    % Returns Network object with rand weights + BIAS for layer a0 to an-1
    net_weights = NetworkParams().rand_init_weights(L_in, L_out);
    
    % Stores Network type weights for all layers
    Network_weights_arr(a) = net_weights;
    
   end
  end
  

  function [hypothesis, activations_arr, sig_activations_arr] = ...
    feedforward(nn_layer_weights, samples, networkModel)
   %%% Feedforwards activations across layers to calulate hypothesis
   % Args:
   %    nn_layers_weights:
   %           NetworkParams arr, contains layer weight matrices
   %    samples: m x n unbiased matrix with sample data
   %    networkModel: [layer1_size, layer2_size,...,layern_size], details
   %                structure of network. 
   % Returns:
   %    hypothesis: array of calculated hypothesis for one forward pass

   assert(networkModel(1) == size(samples,2),...
    'ERROR: input layer size mismatch with sample size')
   
   L_in = samples;
   L_in = ops.add_bias(samples); 
   num_layers = length(networkModel);
   
   activations_arr     = repmat(NetworkParams(), 1, num_layers - 1);
   sig_activations_arr = repmat(NetworkParams(), 1, num_layers - 1);
   
   sig_activations_arr(1) = NetworkParams().append_tensor(L_in); % activation1

   % Forward propagate across rest of activation layers 
   for i = 1:num_layers - 1
    activation = L_in; 
    z = activation * nn_layer_weights(i).layer_weights';
    layer_L_activation = NetworkParams().append_tensor(z);
    activations_arr(i+1) = layer_L_activation; % act_arr(1) is empty
    a = ops.sigmoid(z); % activation
    a = ops.add_bias(a);
    L_out = a; 
    L_in = L_out;
    
    layer_L_sig_activation = NetworkParams().append_tensor(L_out);
    sig_activations_arr(i+1) = layer_L_sig_activation; % sig_act_arr(1) = input layer

   end
   
   L_out = ops.remove_bias(L_out);
   hypothesis = L_out;
   
  end
  
  function [grad_arr] = ...
    backprop_p(hypothesis, weights, acts, sig_acts, Y, lambda, SET)
   
   % Constants
   num_layers = length(SET.networkModel);
   num_hidden = num_layers - 2;
   [m, num_classes] = size(Y); % num_classes true for one-hot Y
   
   % Output layer error
   del_L = hypothesis - Y;
   %imagesc(del_L)
   del_next = del_L;
   
   % Store L layer backprop in obj arr
   del_arr    = repmat(NetworkParams(), 1, num_layers - 1);
   del_arr(1) = NetworkParams().append_tensor(del_L); % del_arr(1) = output layer
   % Calculate del Hadamard product
   for i = 1:num_hidden % Only backprop with g'(x) for hidden layers
    
    del_prev = del_next;
    
    % General formula: del_L = (del_L+1 * W_L) .* g(z_L)
    del_curr = del_prev * weights(num_layers- i ).layer_weights;
    % Apply Hadamard elementwise product of sigmoidGradient
    del_curr = del_curr.*ops.sigmoidGradient(...
     ops.add_bias(acts(num_layers - i ).layer_params));
    
    del_curr = ops.remove_bias(del_curr);
    
    % Append del_L
    del_arr(i + 1) = NetworkParams().append_tensor(del_curr);
    
    del_next = del_curr;
   
     %imagesc(del_curr)
     %pause(0.1)
    
   end
   
   
   % Store L layer errors in obj arr
   delta_arr    = repmat(NetworkParams(), 1, num_layers - 1);

   % Error delta calculations
   for i = 1:num_layers - 1
    % We need two indices to keep backward inverse relationship:
    % "(del_arr(2).layer_params)*(sig_acts(1).layer_params)"
    % "(del_arr(1).layer_params)*(sig_acts(2).layer_params)"
    % "(del_arr(decr).layer_params)*(sig_acts(incr).layer_params)"
    %
    % In general:
    %         del_arr(num_layers - 1)* sig_acts(num_layers - 2)
    %         del_arr(num_layers - 2)* sig_acts(num_layers - 1)
    %         del_arr(num_layers - i++)* sig_acts(num_layers - i--)
    
    incr = i;
    decr = num_layers - i;
    % General formula: delta_L = delta_L + del_L+1 * a_L
    delta_L = del_arr(incr).layer_params' * sig_acts(decr).layer_params; %sum here?
    %size(delta_L)
    delta_arr(decr) = NetworkParams().append_tensor(delta_L);
    
%    imagesc(delta_L)
%    pause(0.1)
    
   end
   
   
   % Store L layer gradients in obj arr
   grad_arr = repmat(NetworkParams(), 1, num_layers - 1);
   C = lambda / m ;
   % Calculate regularized gradients
   for i = 1:num_layers - 1
    
    regularized_weights_L = ...
     C * ops.replace_bias(weights(i).layer_weights);
    
    % Calculate weight_L gradient
    grad_L = 1*(1 / m) * delta_arr(i).layer_params + regularized_weights_L;
    
    
    grad_arr(i) = NetworkParams().append_tensor(grad_L);
    
   end
   
  
   
  end
  
  
  function loss = loss(h, Y)
   %%% Computes unregularized loss for neural network
   % Args: 
   %    h, hypothesis, calculated after feedforward pass
   %    y, labels, target labels matched accordingly 
   %
   % Returns:
   %    loss, double, value of computed loss for that batch (all samples)
   
   [m, ~] = size(Y);
   
   % This expression uses of the Hadamard product '.' of elementwise
   % multiplication
   loss = ( (1/m) * sum( sum( ( -Y.*log(h) - (1-Y).*log(1-h) ), 2) ) );

   
  end
  
  
  function regularization = ...
    regularize(nn_layer_weights, lambda, m, networkModel)
   %%% Computes regularization for neural network
   %    nn_layers_weights:
   %           NetworkParams arr, contains layer weight matrices
   %    lambda: regularization parameter 
   %    m: size of samples
   %    networkModel: [layer1_size, layer2_size,...,layern_size], details
   %                structure of network. 
   % Returns:
   %    regularization: value of regularized constant for cost function
   
   num_layers = length(networkModel); % 3
   
   CONSTANT = ( lambda / (2*m) );
   
   sig = 0;
   for i =1:num_layers - 1
    t = nn_layer_weights(i).layer_weights;
    sig = sig + sum( sum ( t(:, 2:end).^2, 2 ) );
   end
   
   regularization = CONSTANT * sig;
   
  end

  
 end
 
 
end


