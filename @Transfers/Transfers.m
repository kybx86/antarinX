classdef Transfers
 
 methods (Static = true)
  
  function y = sigmoid(tensor)
   % -- Computes sigmoid nonlinear activation
   
   y = 1 ./ (1 + exp(-tensor));
   
  end
  
  function dy = sigmoid_grad(tensor)
   % -- Computes gradient of sigmoid
   
   dy = Transfers.sigmoid(tensor).*(1 - Transfers.sigmoid(tensor));
   
  end
  
  function y = hypertan(tensor)
   % -- Computes tanh nonlinear activation on tensor
   % --  Note: self-defined method 3x slower than matlab tanh(x) function
   
   %  sinh = (exp(tensor) - exp(-tensor)) / 2;
   %  cosh = (exp(tensor) + exp(-tensor)) / 2;
   %  tanh = sinh ./ cosh;
   %  y = tanh;
   
   y = tanh(tensor);
   
  end
  
  function dy = hypertan_grad(tensor)
   % -- Computes gradient of tanh
   
   %dy = 1 - Transfers.hypertan(tensor).^2;
   
   dy = sech(tensor).^2; % use for efficiency
   
  end
  
  function y = softMax(tensor)
   % -- Computes softmax nonlinear activation
   
   [~, n] = size(tensor);
   
   H_shift = tensor - repmat(max(tensor, [], 2), [1, n]);
   y       = exp(H_shift) ./ repmat(sum(exp(H_shift), 2), [1, n]);
   
  end
  
  function dy = softMax_grad(tensor)
   
   % Note: this dy(softmax) currently implemented under Objectives.softmax_loss
   % need one-hot labels
   % need tensor = softmax(x)
   
  end
  
 end
 
end
