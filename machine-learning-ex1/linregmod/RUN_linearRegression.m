%1st layer 

%%documentations etc etc etc explanation of what it does etc. 

close all; clear;
% load raw data
data = load('ex1data2.txt'); 

% create new linear regression model
model = LinearRegressionModel;

% set training model parameters       
model.X                     = data(:, 1:2);
model.y                     = data(:,3); %last column = target label
model.train_data_percentage = .8;
model.bias                  = true;
model.optim_method          = 'BGD'; % documentation here
model.num_iterations        = 100;
model.learning_rate         = 0.01;
model.show_costFunction     = true;
model.show_optimParams      = true;
model.show_data             = true;
model.show_fit              = true;

% initialize training models
deployed_model = initialize(model); % this should return the finished model


%load prediction data
%deployed_model.predict(data)
