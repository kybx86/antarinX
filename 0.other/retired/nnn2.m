function [loss, grad]=nnn2(unrolled_nn_params, samples, labels, lambda, SET)
 

 load('ex4weights.mat');
% 
% % Unroll parameters 
%unrolled_nn_params = [Theta1(:) ; Theta2(:); ];
%SET.networkModel = [400, 128, 32, 10];
%unrolled_fake_params = NetworkParams.fake_unrolled_nnparams(SET.networkModel);
%unrolled_nn_params = unrolled_fake_params;
%subplot(2,1,1)
%plot(unrolled_nn_params, 'r+')
%hold on

rolled_weights_arr = ...
 NetworkParams.roll_nn_params(unrolled_nn_params, SET.networkModel);

% Feedforward  
[hypothesis, activations_arr, sig_activations_arr] = ...
 Network.feedforward(rolled_weights_arr, samples, SET.networkModel);


%imagesc(hypothesis)
 
% Backpropagation
[grad_arr] = Network.backprop_p(hypothesis, ...
 rolled_weights_arr,...
 activations_arr, ...
 sig_activations_arr, ...
 labels,...
 lambda,...
 SET );


%size(grad_arr(1).layer_params)

unrolled_grad = NetworkParams.unroll_nn_params(grad_arr);


%%% cost function for simplicity 
[m, ~] = size(samples);
J = Network.loss(hypothesis, labels);
reg = Network.regularize(rolled_weights_arr, lambda, m, SET.networkModel);
loss = J + reg ; % 117.0255 



grad = unrolled_grad;
%imagesc(grad)
%subplot(2,1,2)
% plot(grad, 'r+')


%bar(loss)
% plot(loss, 'r+') 
% hold on
% pause(0.1)
 

%%%THIS PROVES TO ME THERES AN ERROR IN OUR CODE. 

% fake_grad = NetworkParams.fake_unrolled_nnparams(SET.networkModel);
% fake_grad = rand(1, size(fake_grad, 2));
% grad = fake_grad; 

end

% del_arr(1) = delta3
% sig_acts(1) is biased inputs
% acts(1) is EMPTY since theres no activation from inputs 




%del_arr % 1x2
%size(del_arr(1).layer_params) delta3
%size(del_arr(2).layer_params)  delta 2 5000 25
%sig_acts 1 x3
%size(sig_acts(1).layer_params) % 5000 401
%size(sig_acts(2).layer_params) % 5000 26
%size(sig_acts(3).layer_params) % 5000 11


% delta_arr
% size(delta_arr(1).layer_params) %25 401
% size(delta_arr(2).layer_params) %10 26
% size(delta_arr(3).layer_params) 
