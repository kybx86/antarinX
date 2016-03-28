classdef MinMethods
 
 % -- MinMethods implements various first-order update rules that are commonly
 % -- used for training neural networks. Each update rule accepts current
 % -- weights and the gradient of the loss with respect to those weights
 % -- and produces the next set of weights.
 %
 %  Each update rule has the same interface:
 %
 %            function [next_w, config] = update_func(w, dw, config)
 %
 % Args:
 %     w, matrix, current weights
 %     dw, matrix, of the same shape as w giving the gradient of the loss w.r.t w
 %     config, a struct dictionary containing hyperparameter values such as
 %             learning rate, momentum, etc... If the update rule requires
 %             caching values over many iterations, then config will also hold
 %             these cached values.
 %
 % Note: even if update_func does not use certain hyperparameters, config will
 %      still initalize them, in solver.reset() method
 %
 % Returns:
 %    next_w, The next point after the update.
 %    config, The updated config dictionary to be passed to the next
 %            iteration of theupdate rule.
 %
 % --For efficiency, update rules may perform in-place updates, mutating w and
 %   setting
 
 methods (Static = true)
  
  function [next_w, config] = sgd(w, dw, config)
   
   alpha = config.lr_rate;
   
   next_w = w - alpha*(dw);
   
  end
  
  
  function [next_w, config] = sgd_momentum(w, dw, config)
   
   alpha = config.lr_rate;
   mu    = config.momentum;
   v     = config.velocity;
   
   v = mu*v - alpha*(dw);
   next_w = w + v;
   
   config.velocity = v; % update the param
   
  end
  
  function [next_w, config] = rmsprop(w, dw, config)
   
   %  alpha = config.lr_rate; % alpha = 1e-3 works nice.
   %  decay_rate = config.decay_rate;
   %  epsilon = config.epsilon;
   %  cache = config.cache;
   
   % --Implemeting inplace updates--mutating.
   config.cache = config.decay_rate * config.cache + (1 - config.decay_rate) * dw.^2;
   next_w = w - (config.lr_rate* dw)./ (sqrt(config.cache) + config.epsilon);
   
   %config.cache = cache; % update the param
   
  end
  
 end
 
end
