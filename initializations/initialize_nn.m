%[samples, labels] = ops.load_data('ex2data1.txt');
close all; clear;
%rng('shuffle'); 
rng('default');

% data = load('ex4data1.mat');
% samples = data.X;
% labels = data.y;

data = load('fullMNIST.mat');
samples = data.samples;
labels = data.labels;

% data = load('ex2data2.txt');
% data = data(1:100, :);
% samples = data(:, 1:2);
% labels  = data(:, end);

SET = ParametersModel;
SET.define_trainSplit('train_split', 70, 'Percentage of inputs for training');
SET.define_validationSplit('val_split', 15, 'Percentage of inputs for validation');
SET.define_testSplit('test_split', 15, 'Percentage of inputs for testing');
%SET.define_networkModel('networkModel', [400, 128, 64, 32, 10], 'Network Layer Architecture');
%SET.define_networkModel('networkModel', [784, 256, 64, 10], 'Network Layer Architecture');
SET.define_networkModel('networkModel', [784, 1024, 256, 10], 'Network Layer Architecture');
%SET.define_networkModel('networkModel', [370, 256, 2], 'Network Layer Architecture');
SET.define_lambda('Lambda', 0, 'L2 Regularization value');
SET.define_optimMethod('optim_method', 'sgd', 'Type of update rule');
SET.define_learningRate('alpha', 2, 'Learning Rate');
SET.define_learningRateDecay('lr_decay', 1, 'Learning Rate Step-Decay');
SET.define_annealingFreq('annealing_freq', 3, 'Epoch frquency of lr_decay');
SET.define_momentum('Momentum', 0.9, 'Momentum for optim methods');
SET.define_numEpochs('num_epochs', 10, 'Number of epochs');
SET.define_batchSize('batch_size', 100, 'Size of batch. Note: must fit into data splits');
SET.define_verbose('verbose', true, 'Display output while training');
SET.define_displayEvery('disp_every', 1, 'Plot loss every n epochs');
SET.define_plotLoss('plot_loss', false, 'Plot loss as function of epochs');


disp(SET); disp('Press Enter to Continue ...'); pause; %add a check.
nn_graph(SET, samples, labels);
