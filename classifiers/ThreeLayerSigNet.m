classdef ThreeLayerSigNet < handle
 
 properties
  weights
  lambda
  SET
 end
 
 methods (Static = false)
  
  function obj = ThreeLayerSigNet(~, SET)
   
   obj.weights = NN_utils.nn_weights(SET.networkModel, 'BIAS', 1); %from utils
   obj.lambda  = SET.lambda;
   obj.SET     = SET;
  end
  
  function [loss, grads, hypothesis] = loss(obj, samples, labels, mode)
   
   [m, n] = size(samples);
   samples = ops.add_bias(samples);
   
   % FEEDFORWARD %
   
   [f_out1, cache1] =...
    Affine_Sig.forward('Input', samples, 'Weights', obj.weights{1}, 'AddBias', true);
   
   % fully connected hidden layer 1
   [f_out2, cache2] = ...
    Affine_Sig.forward('Input', f_out1, 'Weights', obj.weights{2}, 'AddBias', true);
   
   % fully connected hidden layer 2
   [f_out3, cache3] = ...
    Affine_Sig.forward('Input', f_out2, 'Weights', obj.weights{3}, 'AddBias', false);
   
   
   hypothesis = f_out3;
   
   
   
   if strcmp(mode, 'test') %string comparison
    hypothesis = f_out3;
    loss = 0;
    grads = {};
    return
   end
   
   
   % BACKPROP %
   
   % fully connected output layer
   [dz4, dy4] = ...
    Affine_Sig.backward_last('Hypothesis', hypothesis, 'Target', labels);
   
   % fully connected hidden layer 2
   [dz3, dy3] = ...
    Affine_Sig.backward('Upstream Derivative', dz4, 'Cache', cache3, 'RemoveBias', true);
   
   % fully connected hidden layer 1
   [dz2, dy2] = ...
    Affine_Sig.backward('Upstream Derivative', dz3, 'Cache', cache2, 'RemoveBias', true);
   
   
   % reg_wN = ( obj.lambda / m) * ops.omit_replace_bias(obj.weights{N});
   reg_weights = Regularizers.regularize_weights(obj.weights, obj.lambda, m);
   
   
   grads{3} = ( (1 / m) * (dz4' * f_out2) ) + reg_weights{3};
   grads{2} = ( (1 / m) * (dz3' * f_out1) ) + reg_weights{2};
   grads{1} = ( (1 / m) * (dz2' * samples) ) + reg_weights{1};
   
   
   %    t1 = grads{1}; t2 = grads{2}; t3 = grads{3} ;
   %    displayData(t1(:, 2:end));
   %    pause(0.001);
   
   cross_entropy = ...
    Objectives.cross_entropy('Hypothesis', hypothesis, 'Target', labels);
   
   L2 = Regularizers.L2(...
    'NetworkWeights', obj.weights, ...
    'Lambda', obj.lambda, ...
    'BatchSize', m, ...
    'NetworkModel', obj.SET.networkModel );
   
   loss = cross_entropy + L2;
   
   
  end
  
 end
 
end