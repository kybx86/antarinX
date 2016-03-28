classdef Affine_Softmax
 
 methods (Static = true)
  
  function [fout, cache] = forward(~, x, ~, w, ~, b)
   % -- Computes forward pass for an affine layer with softMax activation.
   % -- Layer: {Affine -> softMax}.
   %
   % Args:
   %    x, input data, (batch_size x size_layerN), input to layer.
   %    w, weights matrix, (size_layerN+1 x size_layerN), connecting layers L -> L + 1
   %    b, bias, boolean, indicator to add ones bias to fout
   %
   % Returns:
   %    fout, (batch_size x size_layerN+1), matrix of transformed inputs
   %    cache, cell {1x4}...
   %         z: fully-connected layer dot operation
   %         y: nonlinear transfer activation of z
   %         w: (size_layerN+1 x size_layerN), connecting layers L -> L + 1
   %         x: input data, (batch_size x size_layerN), input to layer
   
   z = x * w';
   y = Transfers.softMax(z);
   
   if( b == true )
    y = ops.add_bias(y);
   end
   
   fout = y;
   cache = {z, y, w, x};
   
  end
  
  
  function [dz, dy] = backward_last(~, hypothesis, ~, target)
   % --Computes backward pass for affine with softMax activation OUTPUT layer
   %
   % Args:
   %     hypothesis, (batch_size x K), final transormation on inputs from feedforward
   %     target, (batch_size x K), one-hot matrix of labels
   %
   % Returns:
   %     dy, general derivative w.r.t error function
   %     dz, derivative w.r.t to this layer activation function
   
   % --these are only needed for actual gradient of softmax. So turn off. 
   %[m, n] = size(hypothesis);
   %[~, y_idx] = max(target, [], 2); % capture column indices from Y_hot
   %lin_idx = sub2ind(size(hypothesis), 1:m, y_idx');
   %d_softmax = hypothesis;
   %d_softmax(lin_idx) = d_softmax(lin_idx) - 1;
   %d_softmax = d_softmax / m;
   
   dy = hypothesis - target; % general derivative w.r.t loss function
   dz = (dy);
   
   % --gradient w.r.t to softmax_grad g(x)(1 - g(x))
   %dz = (dy).*Transfers.softMax_grad(hypothesis);
   % --gradient w.r.t to softmax activation (as it should be). 
   %dz = d_softmax;
   % --gradient w.r.t to general loss function (performs better)
   %dz = dy;
   % --note: dz = (dy).*d_softmax does not work. 
   
   
  end
  
 end
 
 
end
