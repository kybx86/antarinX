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
    
   [m, ~] = size(hypothesis);
   
   % --method to calculate dE/dy from labels vector--not needed here. 
   %   note: d_softmax = hypothesis - target.
   %
   %    [~, y_idx] = max(target, [], 2); % capture column indices from Y_hot
   %    lin_idx = sub2ind(size(hypothesis), 1:m, y_idx');
   %    d_softmax = hypothesis;
   %    d_softmax(lin_idx) = d_softmax(lin_idx) - 1;
   %    d_softmax = d_softmax / m;

   
   % --Option #1 (1,1): divide by batch size m, and take grad_softmax
   
   %dy = (hypothesis - target) / m;
   %dz = (dy).*Transfers.softMax_grad(hypothesis); 

   % --results 1: 
   %    method: sgd | lr: 0.1 | 10 epochs: tr_acc = 73.8642, loss = 1.4029
   %    notes: really slow learning but constantly improving for all 10 epochs. 
   % ---------------------------------------------------------------------

   
   % --Option #2 (1 0): divide dy by batch size m, and dont take grad_softmax
   
   %dy = (hypothesis - target) / m;
   %dz = (dy);
   
   % --results 1: 
   %    method: sgd | lr: 0.1 | 10 epochs: tr_acc = 87.621, loss = 0.5923
   %    notes: much faster learning (~75% tr_acc @epoch1) and constantly
   %           improving for all 10 epochs.
   % ---------------------------------------------------------------------    
   
   
   % --Option #3 (0,1): dont divide by batch size, and take grad_softmax
   %   *** this is the one that maintains our API.
   
   dy = hypothesis - target; 
   dz = (dy).*Transfers.softMax_grad(hypothesis);
   
   % --results 1: 
   %    method: sgd | lr: 0.1 | 10 epochs: tr_acc = 83.29, loss = 2.8842
   %    notes: really fast learning (~87% tr_acc @epoch1) but lr maybe (IS) too large!
   %          loss pops after 5 epochs-- best tr_acc = 90.24%
   % --results 2: --lowering lr
   %     method: sgd | lr: 0.01*| 10 epochs: tr_acc = 87.7976, loss = 0.5903
   %     notes: extremely similar to option #2: (~73.9% @ epoch1). 
   %            (Maybe lr_decay needed every 10epochs to further learning)
   % --results 3: --rmsprop (even lower lr)
   %     method: rmsprop | lr: 0.001*| 10 epochs: tr_acc = 90.93, loss=0.8431
   %     notes: needs smaller lr than sgd, (steep loss, and a bit bouncy. lr might 
   %           still be too high since its error increases after epoch5). 
   % --results 4: --rmsprop (2x even lower lr)
   %     method: rmsprop | lr: 0.00001*| 10 epochs: tr_acc = 90.40, loss=0.5035
   %     notes: for lr=1e-4, it bounced off. lr=1e-5 seemed to work better. 
   %            but still slighly bouncy in the loss. tr_acc strady improved! 
   % ---------------------------------------------------------------------
   
   
   % --Option #4 (0,0): dont divide by batch size, and dont take grad_softmax 
   
   %dy = hypothesis - target; 
   %dz = (dy);
   
   % --results 1: 
   %    method: sgd | lr: 0.1 | 10 epochs: tr_acc = 96.56, loss = 0.2936
   %    notes: best learner (~91% tr_acc @epoch1). Learns very quickly(steep loss)
   %    and a bit jumpy which might mean lr a tiny bit too large, but
   %    constantly learns.
   % --results 2: --lowering lr
   %     method: sgd | lr: 0.01*| 10 epochs: tr_acc = 92.3928, loss = 0.4689
   %     notes: similar behaivor to results1, (even with smaller lr), but
   %     learning slows down (although still jumpy). 
   % --------------------------------------------------------------------- 
   
   
  end
  
 end
 
 
end
