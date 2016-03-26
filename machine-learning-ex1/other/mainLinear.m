
%LOADING DATA%
close all;

data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);

self = Hyperparameters;
self.iterations = 100;



%PRE PROCESSING DATA%

%for multi class 
%normalize data when x1 >> xn 

%[model, mu_mean, sigma_std] = normalizeFeatures(model);


% BUILDING MODELS 
model = LinearRegModel(X, y, true);

model = model.addBias;
%model = model.setAlpha(1)

%model = addBias(model, arg1) or model = model.addBias(arg1) are equivalent 

%TRAINING MODELS 
trainer1 = LinearRegTrainer(model, 0.01 , 2000);
trainer2 = LinearRegTrainer(model, 0.01 , 100);




trainer1 = trainer1.gradient_descent;
trainer2 = trainer2.normal_equation;

trainer1.visualize_optimization;
trainer2.visualize_optimization;







