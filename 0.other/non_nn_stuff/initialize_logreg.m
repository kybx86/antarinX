[samples, labels] = ops.load_data('ex2data1.txt');

SET = ModelParameters;
SET.define_trainSplit('train_split', 100, 'Percentage of inputs for training');
SET.define_validationSplit('val_split', 0, 'Percentage of inputs for validation');
SET.define_testSplit('val_split', 0, 'Percentage of inputs for testing');


logistic_regression(SET, samples, labels)