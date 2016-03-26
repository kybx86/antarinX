%2nd layer || model policy 

function deployed_model = initialize(model)

% creates new training model 
trainer_model = TrainerModel(model);

% adds bias to X_tr and X_cv 
if model.bias == true
    trainer_model = trainer_model.addBias;
else
    print('No bias added to data set!')
end


% trains a LinerRegressionModel
switch model.optim_method
    
    case 'BGD'
        trainer_model = trainer_model.trainSGD;
        
    case 'NormalEquation'
        trainer_model = trainer_model.trainNormalEquation;
        
    otherwise
        disp('ERROR: NOT A VALID OPTIMIZIATION METHOD')
end


% shows cost function J vs num_iterations 
if model.show_costFunction == true
    trainer_model.showTrainingCost;
end

% shows contour plot of optimized theta parameters and J surf plot
if model.show_optimParams == true
    trainer_model.showOptimParams;
end

% shows raw** training data X_tr (need to fix this to show just raw
% data
if model.show_data == true
    trainer_model.showTrainingData;
end

% shows linear regression fit to training data X_tr
if model.show_fit == true 
    trainer_model.showFit;
end

%test model
%figure out what to do here. 


%final model
%need to create a new final model here with just the params. 
deployed_model = 0;

end