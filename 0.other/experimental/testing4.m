a = [1:5; 5:9]
b = ones(4,2)
c = 2*ones(2,2)

layer = [400, 25, 10];
num_layers = size(layer, 2);

arr(1)= NetworkWeights().store_reshaped_weights(a);
arr(2) =    NetworkWeights().store_reshaped_weights(b);
arr(3) =    NetworkWeights().store_reshaped_weights(c);

arr(1).layer_weights(:);
arr(2).layer_weights(:) ;

% 
% for i = 1:num_layers - 1 
% 
% %accum = [arr(i).layer_weights(:)]
% next = [arr(i).layer_weights(:) ; arr(i+1).layer_weights(:)]
% accum = [next ; arr(i+1).layer_weights(:)] 
% end
% accum 

   %act_1 = nn_params_model(1).layer_weights(:); %10025 1
   
% i = 1;
% next = [arr(i).layer_weights(:) ; arr(i+1).layer_weights(:)];
% accum = [next ; arr(i+2).layer_weights(:)] ;
% 
%    
% current = [arr(i).layer_weights(:) ; arr(i+1).layer_weights(:)]
% repo = [current ; current]
uni = arr(1).layer_weights(:);
for i = 1:num_layers-1
join = [uni ; arr(i+1).layer_weights(:)]
uni = join;
end

uni 
   
%    tn = [tn ; tn-1]
%    t = [tn ; tn+1]
%    
   
%    num_layers = size(layer, 2); %3
%    %nn_params_model
%    %p = 0;
%    for i = 2:num_layers
%     act_n = nn_params_model(i).layer_weights(:);
%     act_n = layer(i)
%    % p = p + act_n
%    % size(act_n)
%     unrolled_params = [act_1 ; act_n]; %we arrent accumulating 
%     %size(unrolled_params)
%    end
   