classdef Affine_Sig
 
 methods (Static = true)
  
  function [fout, cache] = forward(~, x, ~, w, ~, b)
   % -- Computes forward pass for an affine layer with sigmoid activation.
   % -- Layer: {Affine -> sigmoid}.
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
   y = Transfers.sigmoid(z);
   
   if( b == true )
    y = ops.add_bias(y);
   end
   
   fout = y;
   cache = {z, y, w, x};
   
  end
  
  
  function [dz, dy] = backward(~, dout, ~, cache, ~, b)
   % -- Computes backward pass for an affine layer with sigmoid activation.
   % -- Layer: {Affine -> sigmoid}.
   %
   % Args:
   %    dout, upstream derivative, (size_layerN+1, batch_size)
   %    cache, cell {1x4}, cache from forward pass...
   %         z: fully-connected layer dot operation
   %         y: nonlinear transfer activation of z
   %         w: (size_layerN+1 x size_layerN), connecting layers L -> L + 1
   %         x: input data, (batch_size x size_layerN), input to layer
   %    b, bias, boolean, indicator to add ones bias to fout
   %
   % Returns:
   %     dy, derivative w.r.t to previous layer
   %     dz, derivative w.r.t to this layer activation function
   
   [~, ~, w2, x1] = cache{:};
   
   dy = dout * w2;
   dz = (dy).*Transfers.sigmoid_grad(x1);
   
   if( b == true )
    dz = ops.remove_bias(dz);
   end
   
  end
  
  
  function [dz, dy] = backward_last(~, hypothesis, ~, target)
   % --Computes backward pass for affine with sigmoid activation output layer
   %
   % Args:
   %     hypothesis, (batch_size x K), final transormation on inputs from feedforward
   %     target, (batch_size x K), one-hot matrix of labels
   %
   % Returns:
   %     dy, general derivative w.r.t error function
   %     dz, derivative w.r.t to this layer activation function
   
   dy = hypothesis - target;
   dz = (dy).*Transfers.sigmoid_grad(hypothesis);
   
  end
  
 end
 
 
end
