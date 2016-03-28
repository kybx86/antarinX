classdef TwoLayerSigNet < handle
 
 properties
  weights
  lambda
  SET
 end
 
 methods (Static = false)
  
  function obj = TwoLayerSigNet(~, SET)
   
   obj.weights = NN_utils.nn_weights(SET.networkModel, 'BIAS', 1); %from utils
   obj.lambda  = SET.lambda;
   obj.SET     = SET;
  end
  
  function [loss, grads, hypothesis] = loss(obj, samples, labels, mode)
   % --Architecture: affine - sig -> affine - sig
   
   [m, n] = size(samples);
   
   % --FEEDFORWARD
   
   % input layer
   samples = ops.add_bias(samples);
   
   % fully connected hidden layer 1
   [f_out1, cache1] =...
    Affine_Sig.forward('Input', samples, 'Weights', obj.weights{1}, 'AddBias', true);
   
   % fully connected output layer 2
   [f_out2, cache2] = ...
    Affine_Sig.forward('Input', f_out1, 'Weights', obj.weights{2}, 'AddBias', false);
   
   hypothesis = f_out2;
   
   if strcmp(mode, 'test')
    loss = 0;
    grads = {};
    return
   end
   
   
   % --BACKPROP
   
   % fully connected output layer 3
   [dz3, dy3] = ...
    Affine_Sig.backward_last('Hypothesis', hypothesis, 'Target', labels);
   
   % fully connected hidden layer 2
   [dz2, dy2] = ...
    Affine_Sig.backward('Upstream Derivative', dz3, 'Cache', cache2, 'RemoveBias', true);
   
   % --Compute gradients
   reg_weights = Regularizers.regularize_weights(obj.weights, obj.lambda, m);
   
   grads{2} = ((1 / m) * (dz3' * f_out1)) + reg_weights{2};
   grads{1} = ((1 / m) * (dz2' * samples)) + reg_weights{1};
   
   % --Calculate loss
   cross_entropy = ...
    Objectives.cross_entropy_loss('Hypothesis', hypothesis, 'Target', labels);
   
   L2 = Regularizers.L2(...
    'NetworkWeights', obj.weights, ...
    'Lambda', obj.lambda, ...
    'BatchSize', m, ...
    'NetworkModel', obj.SET.networkModel );
   
   loss = cross_entropy + L2;
   
  end
  
 end
 
end
