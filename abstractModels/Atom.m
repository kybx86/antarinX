classdef Atom 
 properties
  hyperparams
 end
 
 methods (Static = false)
  
  
  function obj = Atom(hyperparams)
   obj.hyperparams = hyperparams;
  end
 end
end