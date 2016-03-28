# antarinX

**antarinX** is a software library for numerical computation and deep learning 
written in Matlab. This library makes use of a highly functional program interpretation 
coupled with an object oriented structure to easily build and train deep neural 
networks. All computations are initialized from initialize_nn.m . This initialization
is passed into the nn_graph.m which creates a computational graph composed of four 
environments: process, inference, swarming, evaluate. Each environment is composed of
a set of nodes particular to that environment: (i.e., process contains nodes related to 
data transformations and pre-processing, whereas inference contains nodes related to 
training the models). Each node in the environment represents mathematical operations, while
graph edges represent objects and tensors that flow between them. 
This enables a clear understanding of the flow of the computations in the graph. 

antarinX was developed by Kevin Yedid-Botton in order to facilitate the creation 
of deep learning and machine learning systems. antarinX is flexible enough to create
a variety of deep learning models such as feedforward networks with any architecture,
but also dynamically structured to facilitate the user experience.


#### *nn_graph example*
```matlab
function nn_graph(SET, samples, labels)

data_model      = process('SET', SET, 'Samples', samples, 'Labels', labels);
[net, solver]   = inference('SET', SET, 'DataModel', data_model);
[~, best_model] = swarming('SolverNN', solver, 'SearchDepth', 0);
best_model = net;
accuracy        = evaluate('DataModel', data_model, 'Model', best_model);

end


function data_model = process(~, SET, ~, samples, ~, labels)

% --One-hot encoding for Y
onehot_labels = ops.dense_to_hot(labels);
% --Creates data_model that randomly* splits data with according to partition
data_model = ops.split_data('Samples', samples, ...
                            'Labels', labels, ...
                            'OneHotLabels', onehot_labels, ...
                            'RandomSel', true,...
                            'ModelConfiguration', SET);
end

function [net, solver] = inference(~, SET, ~, data_model)

% --Creates neural network classifier
net = ThreeLayerTanhNetSoftmax('ModelConfiguration', SET);
% --Creates solver instance protocol
solver = SolverNN('Model', net ,'DataModel', data_model ,'ModelConfiguration', SET);
% --Trains classifier
solver.train();

end

function [best_atom, best_model] = swarming(~, solver, ~, search_depth)

% --Creates swarming instance protocol
atom = Swarming('SolverNN', solver);
% --Performs hyperparamter quasi_grid_search
[best_atom, best_model] = atom.grid_search('SearchDepthPercentage', search_depth);

end

function test_acc = evaluate(~, data_model, ~, net)

% --Compute loss on testing data
[~, ~, hypothesis] = net.loss(data_model.X_tt, data_model.y_tt, 'test');
% --Calculate accuracy from loss
test_acc = ops.calculate_accuracy('Hypothesis', hypothesis, ...
                                  'Target', data_model.yraw_tt, ...
                                  'Verbose', true);
end

```

##For more information

antarinX will serve as the stepping stone to all future developments. 
