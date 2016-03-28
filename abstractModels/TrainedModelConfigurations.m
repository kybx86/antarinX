classdef TrainedModelConfigurations

 properties

  train_split
  val_split
  test_split
  network_model
  L2_reg
  optim_method
  initial_lr
  final_lr
  lr_decay
  annealing_freq
  num_epochs
  num_iter
  batch_size
  sample_size_tr
  initial_loss
  final_loss
  best_tr_acc
  best_cv_acc
  best_hyperparams

 end


 methods (Static = false)

  function obj = TrainedModelConfigurations(solver, SET)


  obj.train_split    = SET.train_split;
  obj.val_split      = SET.val_split;
  obj.test_split     = SET.test_split;
  obj.network_model  = SET.networkModel;
  obj.L2_reg         = SET.lambda;
  obj.optim_method   = SET.optim_method;
  obj.initial_lr     = SET.alpha;
  obj.final_lr       = solver.alpha;
  obj.lr_decay       = SET.lr_decay;
  obj.annealing_freq = SET.annealing_freq;
  obj.num_epochs     = SET.num_epochs;
  obj.num_iter       = solver.num_iter;
  obj.batch_size     = SET.batch_size;
  obj.sample_size_tr = solver.sample_size_tr;
  obj.initial_loss   = solver.loss_history(1);
  obj.final_loss     = solver.loss_history(solver.epoch);
  obj.best_tr_acc    = solver.best_tr_acc;
  obj.best_cv_acc    = solver.best_cv_acc;
  obj.best_hyperparams = solver.best_hyperparams;

  obj.dispSummary();
  %obj.writeSummary(); needs to be fixed for changing networkModel models

  end


  function dispSummary(obj)

   fprintf('\n \n')
   fprintf('\n TRAINING COMPLETE \n');
   fprintf('\n Data Partition \n')
   fprintf('    Training   : %d percent \n', obj.train_split );
   fprintf('    Validation : %d percent \n', obj.val_split );
   fprintf('    Testing    : %d percent \n', obj.test_split );

   fprintf('\n Model Configuration & Optimization \n');
   fprintf('    Network Architecture        : %d \n', obj.network_model );
   fprintf('    Optimization Method         : %s \n', obj.optim_method );
   fprintf('    Initial L2 Regularization   : %.3f \n', obj.L2_reg );
   fprintf('    Initial Learning Rate       : %f \n', obj.initial_lr );
   fprintf('    Final Learning Rate         : %d \n', obj.final_lr );
   fprintf('    Learning Rate Annealing     : %.3f \n', obj.lr_decay );
   fprintf('    Annealing Frequency         : %d \n', obj.annealing_freq );
   fprintf('    Total Number of Epochs      : %d \n', obj.num_epochs );
   fprintf('    Total Number of Iterations  : %d \n', obj.num_iter );
   fprintf('    Batch Size                  : %d \n', obj.batch_size );
   fprintf('    Training Set Data Size      : %d \n', obj.sample_size_tr );

   fprintf('\n Model Accuracy & Performance \n');
   fprintf('    Initial loss             : %.4f \n', obj.initial_loss );
   fprintf('    Final loss               : %.4f \n', obj.final_loss );
   fprintf('    Best Training Accuracy   : %.2f \n', obj.best_tr_acc );
   fprintf('    Best Validation Accuracy : %.2f \n', obj.best_cv_acc );
   fprintf('    Best Hyperparams         : %f \n', obj.best_hyperparams);
   %fprintf('\n Saving Learned Parameters ... "weights.mat" \n');
   %save('weights.mat', 'weights');
   fprintf('\n')
   fprintf('\n END OF TRAINING SUMMARY \n');

  end

  function writeSummary(obj)

   % configured for the above settings and a network of 1 hidden layer
   summary = {obj.train_split, obj.val_split, obj.test_split, ...
    obj.network_model(1), obj.network_model(2), obj.network_model(3), ...
    obj.optim_method, obj.L2_reg, obj.initial_lr, obj.final_lr, obj.lr_decay...
    obj.annealing_freq, obj.num_epochs, obj.num_iter, obj.batch_size, ...
    obj.sample_size_tr, obj.initial_loss, obj.final_loss, obj.best_tr_acc, obj.best_cv_acc};

   fileID = fopen('modelConfigurationsDataBase.csv', 'a');
   formatSpec = '%d, %d, %d, %d, %d, %d, %s, %.3f, %f, %.3f, %d, %d, %d, %d, %d, %.4f, %.4f, %f, %f, %d\n'; %issue with last line
   fprintf(fileID, formatSpec, summary{:});
   fclose(fileID);

  end



 end

end
