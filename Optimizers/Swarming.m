classdef Swarming < handle
 % --  The Swarm class encapsulates the logic to perform a quasi particle
 % -- swarm optimzation (PSO) procedure to exhaust all of the possibilities of
 % -- the best hyperparams--Disclaimer: the swarming method currently does not
 % -- optimize anything. It simply performs a brute force approach to find the
 % -- best hyperparams corresponding to the highest cross-validation accuracy.
 % -- In reality, the hyperparams should be adjusted based on some error given
 % -- by the hyperparams to an error function. However, given the current API,
 % -- it is more computationally expensive to perfom that optimization than
 % -- the brute force approach (although the optimziation might yield better
 % -- hyperparams).
 %
 % -- A atom works on a solver object that must conform to this API:
 %
 %    must contain: solver.train() and solver.resest() methods.
 %    solver must also contain all of the hyperparams properties referenced from
 %    this class.
 %
 %   Example usage:
 %
 %           model = MyAwesomeModel(hidden_size=100, reg=10, ...)
 %           solver = Solver(model, data, ...)
 %           atom = Swarming(mySolver=solver)
 %           best_model = atom.grid_search()
 %
 
 
 properties (Access = public)
  solver
  treshold_epoch
 end
 
 methods (Static = false)
  
  function obj = Swarming(~, solver)
   obj.solver = solver;
  end
  
  function [best_atom, best_model] = grid_search(obj, ~, search_depth)
   % -- grid_search is a exhaustive approach that is computationally
   % -- expensive, but results in the optimal configuration of parameters.
   % -- This should be used when the combinatory space (set1*set2*...setN) is
   % -- small (<3D). Otherwise, other methods (DecisionTrees, Random Search)
   % -- should be used.
   % -- NOTE: this grid_search incorporates a collision_policy to terminate
   % -- solvers that are not learning or are stagnant. This reduces the
   % -- computing time, but the combinatoric space remains the same.
   %
   % Args:
   %     search_depth, scalar, percentage of combinations to explore at random
   %
   % Returns:
   %      best_atom, object containing best hyperparams
   %      best_model, model object trained with best hyperparams
   
   if search_depth == 0
    best_atom = NaN;
    best_model = NaN;
    return
   end
   
   obj.solver.swarming = true; % --apply collider policy & dont write summary
   obj.solver.verbose = true; % --'Epoch n/N | train acc: n | val acc: n | ...'
   obj.solver.collider_policy_verbose = true; % --cause of collision
   obj.solver.treshold_epoch = 10;
   
   ALPHA  = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001];
   LAMBDA = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
   % --shuffle combintatory space
   alpha_space  = length(ALPHA);
   lambda_space = length(LAMBDA);
   alpha_shff   = ALPHA(randperm(alpha_space));
   lambda_shff  = LAMBDA(randperm(lambda_space));
   
   % --initialize atoms
   atoms = [];
   for a = 1 : alpha_space
    for m = 1 : lambda_space
     params = [alpha_shff(a), lambda_shff(m)];
     atoms  = [atoms ; Atom(params)]; % --atom contains one permutation of alpha and lambda
    end
   end
   
   best_atom_cv_acc = 0;
   best_atom_tr_acc = 0;
   sample_space     = length(atoms);
   adj_sample_space = round(sample_space * (search_depth / 100));
   for i = 1 : adj_sample_space
    tStart = tic; % --keep track of elpased time per search
    obj.solver.reset();
    
    params = atoms(i).hyperparams;     % --THIS lr a  lambda b
    obj.solver.alpha      = params(1); % --setting alpha for this atom
    obj.solver.SET.lambda = params(2); % --setting lambda for this atom
    
    % --we create a new net and pass the existing solver.SET with a new value
    % --since lambda is tied to the loss and thus to the network, we need to
    % --create a new instance each time with a different lambda. The solver
    % --does not need to be created since we can modify its alpha from this
    % --class
    switch length(obj.solver.SET.networkModel)
     case 3
      obj.solver.model = TwoLayerSigNet('SET', obj.solver.SET); % need to update to new nets
     case 4
      obj.solver.model = ThreeLayerSigNet('SET', obj.solver.SET);
     case 5
      obj.solver.model = FourLayerSigNet('SET', obj.solver.SET);
    end
    
    % --trains a model given the model above.
    obj.solver.train() % --handle class
    
    atom_tr_acc = obj.solver.tr_acc;
    atom_cv_acc = obj.solver.cv_acc;
    
    if atom_cv_acc > best_atom_cv_acc
     best_atom_cv_acc = atom_cv_acc;
     best_atom = atoms(i);
     best_model = obj.solver.model;
    end
    if atom_tr_acc > best_atom_tr_acc
     best_atom_tr_acc = atom_tr_acc;
    end
    
    t_elapsed = toc(tStart); % --time per iter
    t_tot  = t_elapsed * sample_space;
    t_curr = t_elapsed * i;
    ETAsec = round(t_tot - t_curr);
    ETAmin = ETAsec / 60;
    if ETAsec > 60
     fprintf('\n \t Atom %d/%d/%d |\t Epoch %d/%d |\t ETA %.1f min |\t Collisions: %d',...
      i, adj_sample_space, sample_space, obj.solver.epoch,...
      obj.solver.num_epochs, ETAmin, obj.solver.num_collided);
    else
     fprintf('\n \t Atom %d/%d/%d |\t Epoch %d/%d |\t ETA %.1f sec |\t Collisions: %d',...
      i, adj_sample_space, sample_space, obj.solver.epoch, ...
      obj.solver.num_epochs, ETAsec, obj.solver.num_collided);
    end
   end
   fprintf('\n');
   fprintf('\n Best Model and Hyperparameters \n');
   fprintf('\n \t Alpha: %.3f | Lambda: %.3f | Train acc: %.2f | Val acc: %.2f \n', ...
    best_atom.hyperparams(1), best_atom.hyperparams(2), best_atom_tr_acc, best_atom_cv_acc);
   
  end
  
 end
 
end

% OLD GRID_SEARCH METHOD
%   function swarm(obj)
%    % We only need to do this the first time for a new unseen model
%
%    obj.swarming = true;
%
%    LR_rate = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001];
%    LAMBDA = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
%
%    model_tr_acc_history = zeros(length(LR_rate), length(LAMBDA));
%    model_cv_acc_history = zeros(length(LR_rate), length(LAMBDA));
%    best_model_cv_acc = 0;
%    best_model_tr_acc = 0;
%
%    for r = 1:length(LR_rate)
%     for k = 1:length(LAMBDA)
%
%      obj.reset();
%      obj.SET.lambda = LAMBDA(k);
%      obj.model = TwoLayerNet('SET', obj.SET);
%
%      obj.alpha = LR_rate(r);
%      obj.train(); %optimizes weights with lambda k
%      %obj.model.lambda = 0; %hypothesis does not use lambda (but loss does!)
%
%      model_tr_acc = ...
%       obj.check_accuracy(obj.data_model.X_tr, obj.data_model.y_tr, obj.data_model.yraw_tr, false);
%
%      model_cv_acc = ...
%       obj.check_accuracy(obj.data_model.X_cv, obj.data_model.y_cv, obj.data_model.yraw_cv, false);
%
%      model_tr_acc_history(r, k) = model_tr_acc;
%      model_cv_acc_history(r, k) = model_cv_acc;
%
%      if model_cv_acc > best_model_cv_acc
%       best_model_cv_acc = model_cv_acc;
%       best_hyper_lambda = LAMBDA(k);
%       best_hyper_alpha  = LR_rate(r);
%      end
%      if model_tr_acc > best_model_tr_acc
%       best_model_tr_acc = model_tr_acc;
%      end
%
%      fprintf('\n ')
%     end
%    end
%
%
%    [val, idx] = max(model_tr_acc_history(:));
%    [row, col] = ind2sub(size(model_tr_acc_history), idx);
%    model_cv_acc_history(row, col) = -val;
%    colormap('winter');
%    imagesc(model_cv_acc_history); %plots hyperparam matrix highlighting optim
%    %plot(LAMBDA, model_tr_acc_history, LAMBDA, model_cv_acc_history);
%
%    fprintf('\n Best Model and Hyperparameters \n');
%    fprintf('\n \t Alpha: %f| Lambda: %f| Train acc: %.2f| Val acc: %.2f \n', ...
%     best_hyper_alpha, best_hyper_lambda, best_model_tr_acc, best_model_cv_acc);
%
%   end
%
%
