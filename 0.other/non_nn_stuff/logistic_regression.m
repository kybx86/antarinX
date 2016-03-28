function logistic_regression(SET, samples, labels)

[data_model, num_labels] = inference(SET, samples, labels); 
weights    = training(data_model, num_labels);
accuracy   = evaluate(data_model, weights);

end


function [data_model, num_labels] = inference(SET, samples, labels)

num_labels = ops.count_labels(labels); 

bias_samples = ops.add_bias(samples); 

data_model = ops.split_data(bias_samples, labels, SET); 

end


function weights_tensor = training (data_model, num_labels)

[~ , dim] = size(data_model.X_tr);

weights = ops.new_1D_tensor(dim); 

%loss = ops.binary_logreg_loss(data_model.X_tr, data_model.y_tr, 0);
loss = ops.multi_logreg_loss(data_model.X_tr, data_model.y_tr, 0); 

weights_tensor = Optimizer('Max Iter', 50).multi_min_cg...
 (loss, weights, data_model.y_tr, num_labels);

%weights_tensor = Optimizer('Max Iter', 100).min_cg(loss, weights);

end


function accuracy = evaluate(data_model, weights)

%accuracy = Classifier(data_model).evaluate(weights) %need to fix


accuracy = ops.binary_evaluate(data_model.X_tr, data_model.y_tr, weights);


end

