classdef ParametersModel <handle
 properties
  
  train_split
  val_split
  test_split
  networkModel
  lambda                
  optim_method
  alpha
  lr_decay
  momentum
  momentum_decay
  annealing_freq
  num_epochs
  batch_size
  verbose 
  plot_loss
  plot_interval
  
  max_steps                
  
 end
 
 
 methods (Static = false)
  
  function obj = ParametersModel()
   %tic %start timer
  end
  
  function define_trainSplit(obj, ~, value, ~)
   obj.train_split = value;
  end
  
  function define_validationSplit(obj, ~, value, ~)
   obj.val_split = value;
  end
  
  function define_testSplit(obj, ~, value, ~)
   obj.test_split = value;
  end
  
  function define_maxSteps(obj, ~, value, ~)
   obj.max_steps = value;
  end
  
  function define_lambda(obj, ~, value, ~)
   obj.lambda = value;
  end
  
  function define_optimMethod(obj, ~, value, ~)
   obj.optim_method = value;
  end
  
  function define_learningRate(obj, ~, value, ~)
   obj.alpha = value;
  end
  
  function define_momentum(obj, ~, value, ~)
   obj.momentum = value;
  end
  
  function define_learningRateDecay(obj, ~, value, ~)
   obj.lr_decay = value;
  end 
  
  function define_annealingFreq(obj, ~, value, ~)
   obj.annealing_freq = value;
  end
  
  
  function define_numEpochs(obj, ~, value, ~)
   obj.num_epochs = value;
  end
  
  function define_batchSize(obj, ~, value, ~)
   obj.batch_size = value;
  end
  
  function define_verbose(obj, ~, value, ~)
   obj.verbose = value;
  end
  
  function define_plotLoss(obj, ~, value, ~)
   obj.plot_loss = value;
  end
  
  function define_networkModel(obj, ~, arr, ~)
   obj.networkModel = arr;
  end
  
  function define_output(obj, ~, value, ~)
   obj.output = value;
  end
  
  function define_displayEvery(obj, ~, value, ~)
   obj.plot_interval = value;
  end
  
  
  
  
 end
 
end


