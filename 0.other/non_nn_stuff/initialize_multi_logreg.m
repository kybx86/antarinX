%[samples, labels] = ops.load_data('ex2data1.txt');

data= load('ex3data1.mat');
samples = data.X;
labels = data.y;

SET = ModelParameters;
SET.define_trainSplit('train_split', 100, 'Percentage of inputs for training');
SET.define_validationSplit('val_split', 0, 'Percentage of inputs for validation');
SET.define_testSplit('test_split', 0, 'Percentage of inputs for testing');


multi_logistic_regression(SET, samples, labels)