classdef Sig %NEEDS TO BE REVISED !!
 
 methods (Static = true)
  
  function [fout, cache] = forward(~, z, ~, b)
   %  Computes the forward pass for an affine (fully-connected) layer.
   %
   %  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
   %  examples, where each example x[i] has shape (d_1, ..., d_k). We will
   %  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
   %  then transform it to an output vector of dimension M.
   %
   %  Inputs:
   %  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
   %  - w: A numpy array of weights, of shape (D, M)
   %  - b: A numpy array of biases, of shape (M,)
   %
   %  Returns a tuple of:
   %  - out: output, of shape (N, M)
   %  - cache: (x, w, b)
   
   y = Transfers.sigmoid(z);
   
   if( b == true )
    y = ops.add_bias(y);
   end
   
   fout = y;
   cache = {z, y};
   
  end
  
  
  function [dy] = backward(~, dout, ~, cache, ~, b)
   %   Computes the backward pass for an affine layer.
   %
   %   Inputs:
   %   - dout: Upstream derivative, of shape (N, M)
   %   - cache: Tuple of:
   %     - x: Input data, of shape (N, d_1, ... d_k)
   %     - w: Weights, of shape (D, M)
   %
   %   Returns a tuple of:
   %   - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
   %   - dw: Gradient with respect to w, of shape (D, M)
   %   - db: Gradient with respect to b, of shape (M,)
   %   """
   
   [~, ~, w2, x1] = cache{:};
   
   dy =Transfers.sigmoid_grad(x1);
   
   
   if( b == true )
    dy = ops.remove_bias(dy);
   end
   
  end
  
  
  function [dz, dy] = backward_last(~, hypothesis, ~, target)
   
   dy = hypothesis - target;
   dz = (dy).*Transfers.sigmoid_grad(hypothesis);
   
  end
  
 end
 
 
end