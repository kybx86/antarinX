classdef ThreeLayerTanhNet < handle
 
 properties
  weights
  lambda
  SET
 end
 
 methods (Static = false)
  
  function obj = ThreeLayerTanhNet(~, SET)
   
   obj.weights = NN_utils.nn_weights(SET.networkModel, 'BIAS', 1); %from utils
   obj.lambda  = SET.lambda;
   obj.SET     = SET;
  end
  
  function [loss, grads, hypothesis] = loss(obj, samples, labels, mode)
   
   % Architecture: affine - tanh -> affine - tanh -> affine - tanh
   
   [m, ~] = size(samples);
   
   % --FEEDFORWARD
   
   % input layer
   samples = ops.add_bias(samples);
   
   % fully connected hidden layer 1
   [f_out1, ~] =...
    Affine_Tanh.forward('Input', samples, 'Weights', obj.weights{1}, 'AddBias', true);
   
   % fully connected hidden layer 2
   [f_out2, cache2] = ...
    Affine_Tanh.forward('Input', f_out1, 'Weights', obj.weights{2}, 'AddBias', true);
   
   % fully connected output layer 3 // softmax better than this
   [f_out3, cache3] = ...
    Affine_Tanh.forward('Input', f_out2, 'Weights', obj.weights{3}, 'AddBias', false);
   
   hypothesis = f_out3;
   
   if strcmp(mode, 'test')
    hypothesis = f_out3;
    loss = 0;
    grads = {};
    return
   end
   
   % --BACKPROP
   
   % fully connected output layer 3
   [dz4, ~] = ...
    Affine_Tanh.backward_last('Hypothesis', hypothesis, 'Target', labels);
   
   % fully connected hidden layer 2
   [dz3, ~] = ...
    Affine_Tanh.backward('Upstream Derivative', dz4, 'Cache', cache3, 'RemoveBias', true);
   
   % fully connected hidden layer 1
   [dz2, ~] = ...
    Affine_Tanh.backward('Upstream Derivative', dz3, 'Cache', cache2, 'RemoveBias', true);
   
   % --Compute gradients
   reg_weights = Regularizers.regularize_weights(obj.weights, obj.lambda, m);
   
   grads{3} = ((1 / m) * (dz4' * f_out2))  + reg_weights{3};
   grads{2} = ((1 / m) * (dz3' * f_out1))  + reg_weights{2};
   grads{1} = ((1 / m) * (dz2' * samples)) + reg_weights{1};
   
   % --Calculate cost
   
   cross_entropy_adj = ...
    Objectives.cross_entropy_adjusted('Hypothesis', hypothesis, 'Target', labels);
   
   L2 = Regularizers.L2(...
    'NetworkWeights', obj.weights, ...
    'Lambda', obj.lambda, ...
    'BatchSize', m, ...
    'NetworkModel', obj.SET.networkModel );
   
   loss = cross_entropy_adj + L2;
   
  end
  
 end
 
end
