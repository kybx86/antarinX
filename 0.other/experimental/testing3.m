%layer = [80, 45, 5, 2];
%layer = [400, 25, 10];
%layer = [400, 128, 32, 10];

   act_1 = nn_params_model(1).layer_weights(:); %10025 1
   num_layers = size(nn_params_model, 2); %3

   for i = 2:num_layers
    act_n = nn_params_model(i).layer_weights(:);
   % p = p + act_n
   % size(act_n)
    unrolled_params = [act_1 ; act_n]; %we arrent accumulating 
    %size(unrolled_params)
   end
   
   




unrolled_fake_params = NetworkWeights.fake_unrolled_nnparams(layer);

rolled_params_tensor = ...
 NetworkWeights.roll_nn_params(unrolled_fake_params, layer)

%success . 





% %nn_params_model = NetworkWeights.nn_weights(layer);
% 
% 
% %utility function to create fake unrolled weights for net of any size
% 
% BIAS = 1;
% T = 0;
% 
% for i = 1:length(layer)-1
%  
%  L = ( ( layer(i) + BIAS ) * layer(i+1) ); %400+1 * 25
%  %R = ( ( layer(2) + BIAS ) * layer(3) ) %25+1 * 10
%  
%  T = T + L;
%  
% end
% 
% 
% params = ones(T, 1); % 10285 1
% size(params);
% 
% 
% 
% adj = 1;
% 
% %first weights
% w0 = reshape(params(1 : ( layer(2)*(layer(1)+BIAS) ) ), [layer(2), layer(1)+BIAS]);
% size(w0) %25 x401
% %length(w0)
% layer0_weights = NetworkWeights().store_reshaped_weights(w0);
% Network_weights_arr(1) = layer0_weights;
% %rest of weights
% num_layers = length(layer);
% 
% for i = 1:num_layers- 2
%  
%  prev = ( ( layer(i+1) * (layer(i)+BIAS) ) + adj ) ;
%  
%   layerN_weights = reshape( params( ( ( layer(i+1) * (layer(i)+BIAS) ) + adj ) :...
%   prev + ( ( layer(i+2) * ( layer(i+1)+BIAS ) ) - adj  ) ),...
%   [layer(i+2), (layer(i+1)+BIAS) ] ) ;
%  
%  size( layerN_weights);
%  
%  layerN_weights = NetworkWeights().store_reshaped_weights(layerN_weights);
%  
%  Network_weights_arr(i+1) = layerN_weights;
%  
% end
% 



%size of W will always be L_out * L_in+bias
