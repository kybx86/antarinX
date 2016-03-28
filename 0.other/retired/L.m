function [Theta1, Theta2] = L(samples, labels, nn_params_model, lambda, SET)

samples = ops.add_bias(samples);

Theta1 = nn_params_model(1).layer_weights;
Theta2 = nn_params_model(2).layer_weights;
%load('ex4weights.mat');
Y = labels;
m = size(labels,1);
num_iter = 1e3;
loss_hist = zeros(1, num_iter);

for i=1: num_iter;
 
 % fully connected input layer 
 [f_out1, cache1] = ...
  NN.affine_sig_forward('Input', samples, 'Weights', Theta1, 'AddBias', true);
 
 % fully connected hidden layer 
 [f_out2, cache2] = ...
  NN.affine_sig_forward('Input', f_out1, 'Weights', Theta2, 'AddBias', false);
 
 hypothesis = f_out2;
 
 
 % fully connected output layer
 [dz3, dy3] = ...
  NN.affine_sig_backward_last('Hypothesis', hypothesis, 'Target', labels);
 
 % fully connected hidden layer
 [dz2, dy2] = ...
  NN.affine_sig_backward('Upstream Derivative', dz3, 'Cache', cache2, 'RemoveBias', true);
 
 
 cross_entropy = NN.cross_entropy_loss('Hypotheis', hypothesis, 'Target', Y);
 
 L2 = ...
  NN.L2_regularization('NN Weights', nn_params_model, ...
  'Lambda', lambda, ...
  'BatchSize', m, ...
  'NetworkModel', SET.networkModel);
 
 
 
 
 loss = cross_entropy + L2;
 loss_hist(i) = loss;
 
 if mod(i, 100) == 0
  fprintf('\n loss at %d : %f ', i, loss)
  plot(loss_hist(1:i))
  drawnow;
 end
 
 
 
 %SGD
 
 regularizedTheta2 = (lambda/m) * [zeros(size(Theta2, 1), 1), Theta2(:,2:end)]; 
 regularizedTheta1 = (lambda/m) * [zeros(size(Theta1, 1),1), Theta1(:,2:end)]; %25 401

 %
 % Theta1_grad = (1/m) * del_1 + regularizedTheta1;
 % Theta2_grad = (1/m) * del_2 + regularizedTheta2;
 
 
 Theta2 = Theta2 - 0.1*( ( (dz3'*f_out1) / m ) + regularizedTheta2);
 Theta1 = Theta1 - 0.1*( ( (dz2'*samples) / m ) + regularizedTheta1);
%  displayData(Theta1(:, 2:end));
%  pause(0.0001)
 
end
%subplot(2,1,2)
%plot(loss)



end

%%I JUST WANT THIS FUNCTION TO BE THE LOSS AND THE GRADS SO THAT THE SOLVER
%%CAN ACTUALLY ITERATE OVER IT. 


