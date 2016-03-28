classdef NetworkParams %% ALL OF THIS SHOULD GO INTO NN_UTILS 
 
 properties 
 % each NetworkParams object corresponds to 1 connecting layer 
 layer_weights 
 layer_params
 end
 
 
 methods (Static = false)
 
  function obj = NetworkParams()
   % Empty constructor 
  end
  
  function obj = rand_init_weights(obj, L_in, L_out)
   %%% Random initialization of layer a weights
   % Args:
   %    int: L_in number of incoming connections 
   %    int: L_out number of outgoing connections 
   % Returns:
   %    NetworkWeight object with initialized 'layer_weights'
   
   % Note: The first row of W corresponds to  the bias units
   BIAS = 1;
   epsilon = sqrt(6)/sqrt(L_in + L_out);
   obj.layer_weights = rand(L_out, BIAS + L_in) * (2*epsilon) - epsilon;
   
   
  end

  function obj = store_reshaped_weights(obj, nlayer_weight)
   %%% Stores weight matrices in obj array
   % Args:
   %    matrix: reshaped weight for n layer
   % Returns:
   %    NetworkWeight object with 'layer_weights'
   
   obj.layer_weights = nlayer_weight;
   
  end
  
    function obj = append_tensor(obj, nlayer_params)
   %%% Stores weight matrices in obj array
   % Args:
   %    matrix: reshaped weight for n layer
   % Returns:
   %    NetworkWeight object with 'layer_weights'
   
   obj.layer_params = nlayer_params;
   
  end
  
 end
%  
%  methods (Static = true)
%   
%   function Network_weights_arr = nn_weights(net_layers)
%    %%% Creates layer weights for network of any size
%    % Args:
%    %    array: net_layers, [size_layer1, size_layer2 , ... ,size_layerN]
%    %    NOTE: size of layers must be UNBIASED 
%    % Returns:
%    %    NetworkParams array: array of weights for all layers in nn
%    
%    
%    num_layers =  length(net_layers);
% 
%    % Pre-allocation of Network type array
%    Network_weights_arr = repmat(NetworkParams(), 1, num_layers);
%    
%    for a = 1:num_layers - 1 %there will be one weight less than layers
%     
%     L_in   = net_layers(a); 
%     L_out  = net_layers(a+1);
%     
%     % Returns Network object with rand weights + BIAS for layer a0 to an-1
%     net_weights = NetworkParams().rand_init_weights(L_in, L_out); 
%     
%     % Stores Network type weights for all layers
%     Network_weights_arr(a) = net_weights;
%         
%    end
%   end
%     
%   function unrolled_weights = unroll_nn_weights(nn_params_model)
%    %%% Unrolls matrices of size m x n into singular concat column vector
%    %%% (m1*n1) + (m2*n2)... + (m*n)
%    % Args:
%    %    NetworkParams array: layer_weights, matrices of different sizes
%    % Returns:
%    %    Unrolled_params: unrolled column vector containing all weights 
%    
%    num_layers = size(nn_params_model, 2); %
%    
%    uni = nn_params_model(1).layer_weights(:);
%    for i = 1:num_layers - 1
%     join = [uni ; nn_params_model(i+1).layer_weights(:)];
%     uni = join;
%    end
%    
%    unrolled_weights = uni;
%    
%   end
%   
%   function unrolled_params = unroll_nn_params(nn_params_model)
%    %%% Unrolls matrices of size m x n into singular concat column vector
%    %%% (m1*n1) + (m2*n2)... + (m*n)
%    % Args:
%    %    NetworkParams array: layer_params, matrices of different sizes
%    % Returns:
%    %    Unrolled_params: unrolled column vector containing all params
%    
%    num_layers = size(nn_params_model, 2); %
%    
%    uni = nn_params_model(1).layer_params(:);
%    for i = 1:num_layers - 1
%     join = [uni ; nn_params_model(i+1).layer_params(:)];
%     uni = join;
%    end
%    
%    unrolled_params = uni;
%    
%   end
%   
%   function rolled_params_tensor =...
%     roll_nn_params(unrolled_nn_params, networkModel)
%    %%% Rolls single unrolled column vector containing all weights back
%    %%% into weight matrices according to network model layers
%    % Args:
%    %    unrolled_nn_params: single 1D column vector containing all weights
%    %    networkModel: [layer1_size, layer2_size, ..., layern_size], details
%    %                structure of network. 
%    % Returns:
%    %    rolled_params_tensor: NetworkParams arr containg reshaped weights
%    
%    
%    
%    layer = networkModel;
%    num_layers = length(layer);
%    
%    Network_weights_arr = repmat(NetworkParams(), 1, num_layers-1);
%    BIAS = 1;
%    ADJ = 1;
%    
%    % Rolls flattened weights for FIRST layer back into matrix
%    weight_layer_1 = reshape(...
%     unrolled_nn_params(...
%     1 : ( layer(2)*(layer(1)+BIAS) ) ), [layer(2), layer(1)+BIAS]);
%    weight_layer_1 =...
%     NetworkParams().store_reshaped_weights(weight_layer_1);
%    Network_weights_arr(1) = weight_layer_1;
%    
%    % Rolls rest of flattened weights back into matrices stored in obj arr
%    for i = 1:num_layers - 2
%     
%     prev = ( ( layer(i+1) * (layer(i)+BIAS) ) + ADJ ) ;
%     
%     layerN_weights = reshape(...
%      unrolled_nn_params( ( ( layer(i+1) * (layer(i)+BIAS) ) + ADJ ) :...
%      prev + ( ( layer(i+2) * ( layer(i+1)+BIAS ) ) - ADJ  ) ),...
%      [layer(i+2), (layer(i+1)+BIAS) ] ) ;
%     
%     layerN_weights =...
%      NetworkParams().store_reshaped_weights(layerN_weights);
%     
%      Network_weights_arr(i+1) = layerN_weights;
%     
%    end
%    
%    rolled_params_tensor = Network_weights_arr;
%    
%   end 
%   
%   function unrolled_fake_params = fake_unrolled_nnparams(layer)
%     % Utility function to create fake unrolled weights for net of any size
%     % returns unrolled nn_params of ones/zeros
%     
%    num_layers = length(layer) ; 
%    BIAS = 1;
%    accum = 0;
%    
%    for i = 1:num_layers - 1
%     % Always true that eq = (size_of_layer1 + bias) * size_of_layer2 
%     eq = ( ( layer(i) + BIAS ) * layer(i+1) );    
%     accum = accum + eq;
%    end
%    
%    unrolled_fake_params = 0*ones(accum, 1); 
%    
%    
%   end
%   
%  end
   
 
end

