
data= load('ex4data1.mat');
samples = data.X;
labels = data.y;

%layers = [400, 128, 32, 10];
layers = [400, 25, 10];

%creates weights for network of any size
for a = 1:length(layers) - 1 %there will be one weight less than layers
 
 L_in   = layers(a);
 L_out  = layers(a+1);
 
 %returns Network object with rand weights + bias for layer a
 net_weights = Network().rand_init_weights(L_in, L_out); %returns obj 25 401
 
 
 %stores network object weights for all layers in Network Array 
 Network_weights_arr(a) = net_weights;
 
 size(Network_weights_arr(a).weights)
 
end



samples = ops.add_bias(samples);

L_in = samples; 




%%feedforward 

% need raw inputs network weights arr

L_in = ones(5000, 401); % samples MUST contain bias already (eh why)

%function hypothesis = feedforward(inputs)

for i = 1:length(layers)-1
 
 activation = L_in; %m 401
 z = activation * Network_weights_arr(i).weights'; %25
 z = ops.sigmoid(z);
 z = ops.add_bias(z); % m 26
 L_out = z;
 L_in = L_out;  
end

L_out = ops.remove_bias(L_out);
hypothesis = L_out ; 


size(L_out)
imagesc(L_out)



% a1 = L_in 
% z2 = a1* weight1';
% a2 = sigmoid(z2)
%bias 