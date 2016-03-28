function nn_graph(SET, samples, labels)

data_model      = process('SET', SET, 'Samples', samples, 'Labels', labels);
[net, solver]   = inference('SET', SET, 'DataModel', data_model);
[~, best_model] = swarming('SolverNN', solver, 'SearchDepth', 0);
best_model = net;
accuracy        = evaluate('DataModel', data_model, 'Model', best_model);

end


function data_model = process(~, SET, ~, samples, ~, labels)

% One-hot encoding for Y
onehot_labels = ops.dense_to_hot(labels);

% Creates data_model that randomly* splits data with according to partition
data_model = ops.split_data('Samples', samples, ...
                            'Labels', labels, ...
                            'OneHotLabels', onehot_labels, ...
                            'RandomSel', true,...
                            'ModelConfiguration', SET);
end

function [net, solver] = inference(~, SET, ~, data_model)

% Creates neural network classifier
%net = TwoLayerTanhNet('ModelConfiguration', SET);
%net = ThreeLayerTanhNet('ModelConfiguration', SET);
net = ThreeLayerTanhNetSoftmax('ModelConfiguration', SET);
%net = ThreeLayerSigNet('ModelConfiguration', SET);
%net = FourLayerTanhNet('ModelConfiguration', SET);

% Creates solver instance protocol
solver = SolverNN('Model', net ,'DataModel', data_model ,'ModelConfiguration', SET);
% Trains classifier
solver.train();
end

function [best_atom, best_model] = swarming(~, solver, ~, search_depth)

% Creates swarming instance protocol
atom = Swarming('SolverNN', solver);
% Performs hyperparamter quasi_grid_search
[best_atom, best_model] = atom.grid_search('SearchDepthPercentage', search_depth);
end

function test_acc = evaluate(~, data_model, ~, net)

% Compute loss on testing data
[~, ~, hypothesis] = net.loss(data_model.X_tt, data_model.y_tt, 'test');
% Calculate accuracy from loss
test_acc = ops.calculate_accuracy('Hypothesis', hypothesis, ...
                                  'Target', data_model.yraw_tt, ...
                                  'Verbose', true);
end
