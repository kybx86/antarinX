classdef SolverNN < handle
 % -- SolverNN encapsulates all the logic necessary for training classification
 % -- models. SolverNN performs stochastic gradient descent using different
 % -- update rules defined in MinMethods.m
 % -- After the train() method returns, model.weights will be the optimized
 % -- parameters after training.
 % -- SolverNN is also able to perform hyperparameter search called from Swarming
 % -- when obj.swarming is true, SolverNN will apply a collider policy to eliminate
 % -- models with hyperparameters that lead to NaN or Inf loss. Implemented, but
 % -- not in use, is also a moving accuracy avg that collides models that remain
 % -- stagnant in learning. However, this has often led to premature collision.
 %
 % -- SolverNN works on a model object that must conform to the following API:
 %
 %    model.loss(samples, labels, mode):
 %         must be a function that computes training-time loss and
 %         gradients, and test-time classification scores, with the following
 %         inputs and outputs:
 %
 %     Args:
 %         samples, minibatch matrix of input data of shape (batch_size x n).
 %         labels, minibatch matrix of onehot_labels (batch_size x n).
 %         mode, boolean, if test -> only feedforward
 %                        if train -> feedforward & backprop
 %
 %    Returns:
 %        if mode is test, run a test-time forward pass and return:
 %        hypothesis, matrix of shape (batch_size, K)
 %
 %        if mode is train, run a training time forward and backward pass and return:
 %        loss, scalar giving the loss
 %        grads, cell{1 x layers-1}, graients w.r.t to each layer
 %
 properties ( Access = public)
  
  model
  data_model
  optim_method
  alpha
  lr_decay
  annealing_freq
  momentum
  num_epochs
  num_iter
  batch_size
  verbose
  plot_loss
  num_weights
  num_layers
  loss_history
  update_func
  plot_interval
  epoch
  sample_size_tr
  cv_acc
  tr_acc
  tr_acc_history
  cv_acc_history
  best_cv_acc
  best_tr_acc
  best_hyperparams %unsued
  labels
  SET
  configs
  container
  treshold_epoch
  swarming
  num_collided
  collider_policy_verbose
  
  
 end
 
 methods (Static = false)
  
  function obj = SolverNN(~, model, ~, data_model, ~, SET)
   
   % --object properties
   obj.model          = model;
   obj.data_model     = data_model;
   obj.SET            = SET;
   % --configurations
   obj.optim_method   = SET.optim_method;
   obj.alpha          = SET.alpha;
   obj.lr_decay       = SET.lr_decay;
   obj.annealing_freq = SET.annealing_freq;
   obj.momentum       = SET.momentum;
   obj.num_epochs     = SET.num_epochs;
   obj.batch_size     = SET.batch_size;
   obj.verbose        = SET.verbose;
   obj.plot_interval  = SET.plot_interval;
   obj.plot_loss      = SET.plot_loss;
   obj.sample_size_tr = size(data_model.X_tr, 1);
   obj.num_weights    = length(obj.model.SET.networkModel) - 1;
   obj.num_layers     = obj.num_weights + 1;
   % --swarming properties
   obj.swarming       = false;
   obj.num_collided   = 0;
   obj.collider_policy_verbose = false;
   % --MinMethods dictionary
   obj.configs = struct('lr_rate', obj.alpha,...
    'momentum', obj.momentum,...
    'decay_rate', 0.99,...
    'epsilon', 1e-4,...
    'velocity', [],... % velocity is non default value
    'cache', [] );     % cache is non default value
   
   
   try
    method_name = strcat('MinMethods.', SET.optim_method);
    obj.update_func = str2func(method_name);
    obj.update_func(0, 0, obj.configs); %test it works
   catch
    error('Error: Invalid optim_method');
   end
   
   obj.reset()
   
  end
  
  
  function reset(obj)
   
   obj.epoch = 0;
   obj.cv_acc = 0;
   obj.tr_acc = 0;
   obj.best_cv_acc = 0;
   obj.best_tr_acc = 0;
   obj.loss_history = [];
   obj.tr_acc_history = [];
   obj.cv_acc_history = [];
   
   % initializing configuration container
   obj.container = cell(1, obj.num_weights); % contains configs for each layer
   
   for i = 1 : obj.num_weights
    [m, n] = size(obj.model.weights{i});
    obj.configs.velocity = zeros(m, n);
    obj.configs.cache = zeros(m, n);
    obj.container{i} = obj.configs;
   end
   
  end
  
  
  function loss = step(obj)
   
   batch_mask = randperm(obj.sample_size_tr, obj.batch_size);
   X_batch = obj.data_model.X_tr(batch_mask, :);
   y_batch = obj.data_model.y_tr(batch_mask, :);
   
   [loss, grads, ~] = obj.model.loss(X_batch, y_batch, 'train');
   
   % --updates gradients for each layer
   for i = 1 : obj.num_weights
    config = obj.container{i};
    dw     = grads{i};
    w      = obj.model.weights{i};
    [next_w, next_config] = obj.update_func(w, dw, config); % abstract function
    obj.model.weights{i} = next_w;
    obj.container{i} = next_config;
   end
   
  end
  
  
  function acc = check_accuracy(obj, samples, onehot_labels, labels, sampling)
   
   m_set          = size(samples, 1);
   num_subsamples = m_set;
   % --subsampling to a size < m_set is more efficient, but gives stochastic accuracy
   % --i.e. num_subsamples = 1000
   
   % --subsample if samplesize greater than subsamplesize and sampling true
   if (sampling == true) && (m_set > num_subsamples)
    m_set  = num_subsamples;
    mask   = randperm(m_set, num_subsamples);
    X      = samples(mask, :);
    y      = onehot_labels(mask, :);
    y_true = labels(mask, :);
    % [~, y_true] =  max(Y, [], 2); % also works and dont need label vector
    % y_true = y_true(mask,:);
   else
    X      = samples;
    y      = onehot_labels;
    y_true = labels;
   end
   
   % --compute accuracy in batches
   num_batches = (m_set / obj.batch_size);
   assert(num_batches >= 1 && mod(num_batches, 1) == 0,...
    'ERROR: sampling_size less than batch_size AND does not fit evenly in set!');
   if mod(m_set, obj.batch_size) ~= 0
    num_batches = num_batches + 1;
   end
   
   y_pred = [];
   for i = 1 : num_batches
    start  = (i * obj.batch_size) - obj.batch_size + 1;
    finish = ((i + 1) * obj.batch_size) - obj.batch_size;
    [~, ~, hypothesis] = ...
     obj.model.loss( X(start:finish, :), y(start:finish, :), 'test');
    [~, pred] = max(hypothesis, [], 2);
    y_pred = [y_pred; pred];
   end
   
   acc = ( mean(y_pred == y_true) * 100 );
   
  end
  
  
  function train(obj)
   % --run optimization to train the model
   
   assert(obj.sample_size_tr >= obj.batch_size, ...
    'ERROR: sample_size greater than batch_size');
   
   iters_per_epoch  =  ceil(obj.sample_size_tr / obj.batch_size);
   
   assert(mod(iters_per_epoch, 1) == 0,...
    'ERROR: Batch_size and data partition do not split evently:');
   
   obj.num_iter     = ( obj.num_epochs * iters_per_epoch );
   obj.loss_history = zeros(1, obj.num_epochs);
   
   
   % main trainer loop
   for i = 1 : obj.num_iter
    loss = obj.step();
    
    epoch_end = mod( i, iters_per_epoch) == 0;
    if epoch_end
     obj.epoch = obj.epoch + 1;
     if mod(obj.epoch, obj.annealing_freq) == 0
      for k = 1 : length(obj.container)
       %--decay lr_rate from within container
       obj.container{k}.lr_rate = obj.container{k}.lr_rate * obj.lr_decay;
      end
     end
    end
    
    
    first_iter = ( i == 1 );
    last_iter  = ( i == obj.num_iter );
    
    % --check only at first_iter, last_iter, or epoch_end
    if (first_iter || last_iter || epoch_end)
     obj.loss_history(1, obj.epoch + 1) = loss;
     
     obj.tr_acc = obj.check_accuracy(obj.data_model.X_tr,...
      obj.data_model.y_tr, obj.data_model.yraw_tr, true);
     
     obj.cv_acc = obj.check_accuracy(obj.data_model.X_cv, ...
      obj.data_model.y_cv, obj.data_model.yraw_cv, false);
     
     obj.tr_acc_history(1, obj.epoch + 1) = obj.tr_acc; % appending before
     obj.cv_acc_history(1, obj.epoch + 1) = obj.cv_acc;
     
     if obj.verbose % --prints @ every epoch
      fprintf('\n Epoch %d/%d |\t', obj.epoch, obj.num_epochs);
      fprintf('train acc: %f |\t val acc: %f |\tloss at %d: %f',...
       obj.tr_acc, obj.cv_acc, i, loss);
     end
     if (obj.plot_loss==true) && (mod(obj.epoch, obj.plot_interval) == 0)
      plot(obj.loss_history(1:obj.epoch + 1), 'b');
      drawnow;
     end
     if obj.cv_acc > obj.best_cv_acc % --keep track of best_cv_acc
      obj.best_cv_acc = obj.cv_acc;
     end
     if obj.tr_acc > obj.best_tr_acc % --keep track of best_tr_acc
      obj.best_tr_acc = obj.tr_acc;
     end
    end
    % --default set to false. Only applied from Swarming.m
    if obj.swarming % called in each iter
     collide = obj.collider_policy(loss, obj.cv_acc);
     if collide
      break
     end
    end
    
   end
   
   % --when done training, write summary of solver to TrainedModelConfigurations
   if ~obj.swarming
    TrainedModelConfigurations(obj, obj.SET);
   end
   
  end
  
  
  function collide_atom = collider_policy(obj, loss, cv_acc)
   
   collide_atom = false;
   epoch_tol = obj.treshold_epoch; % --tolercance of cv_acc stagnation
   
   if (loss == Inf) || isnan(loss)
    collide_atom = true;
    obj.num_collided = obj.num_collided + 1;
    if obj.collider_policy_verbose
     fprintf('\n');
     fprintf('\n >> Collided Trainer << \n');
     fprintf('\t CAUSE: cross_entropy increase \n');
     fprintf('\t Alpha: %.3f | Lambda: %.3f \n', obj.alpha, obj.model.SET.lambda);
    end
    %    elseif mod(obj.epoch + 1, obj.treshold_epoch) == 0
    %     prev_epoch_tol_avg = mean(...
    %       obj.cv_acc_history(((obj.epoch + 2) - epoch_tol) : (obj.epoch + 0)));
    %     if cv_acc < prev_epoch_tol_avg  % '=' makes big differnce
    %      collide_atom = true;
    %      obj.num_collided = obj.num_collided + 1;
    %      if obj.collider_policy_verbose
    %       fprintf('\n');
    %       fprintf('\n >> Collided Trainer << \n');
    %       fprintf('\t CAUSE: cv_acc stagnant on last %d epochs \n', epoch_tol);
    %       fprintf('\t Alpha: %.3f | Lambda: %.3f \n', obj.alpha, obj.model.SET.lambda);
    %      end
    %     end
   else
    collide_atom = false;
   end
   
  end
  
  
 end
end
