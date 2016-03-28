classdef Affine % not correctly integrated to graph
 
 methods (Static = true)
  
  function [fout, cache] = forward(~, x, ~, w, ~, b)
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
   
   z = x * w';
   y = z;
   
   if( b == true )
    y = ops.add_bias(y);
   end
   
   fout = y;
   cache = {z, y, w, x};
   
  end
  
  
  function [dz, dy] = backward(~, dout, ~, cache, ~, b)
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
   
   [~, y, w2, x1] = cache{:};
   
   %    disp('dout');size(dout) % 100, 10
   %    disp('w2');size(w2) % 10 257
   %    disp('x1');size(x1) % 100 257
   %    disp('y');size(y) %100 10
   %
   %    dy = y*w2;
   %    %dz = (dout).*(dy);
   %    disp('dy'); size(dy); %100 257
   %    dz = dy;
   %    disp('dz'); size(dz)
   %    dz = dout.*
   %    dz = (dy).*Transfers.hypertan_grad(x1);
   %    dy = dout ; %dout*w2; %100 257
   %    dz = (dy);%.*(x1); % not sure here.
   %
   %    dy = dout*w2; % dout = y-t
   %    dz = (dy).*Transfers.hypertan_grad(x1);
   %
   %    dy = dout ;%hypothesis - target; % general derivative w.r.t loss function
   %    dz = (dy).*Transfers.hypertan_grad(y);
   dy = dout;
   dz = dy.*y;
   
   
   if( b == true )
    dz = ops.remove_bias(dz);
   end
   
  end
  
  
  function [dz, dy] = backward_last(~, hypothesis, ~, target)
   
   dy = hypothesis - target;
   dz = (dy).*(hypothesis);
   
  end
  
 end
 
 
end
