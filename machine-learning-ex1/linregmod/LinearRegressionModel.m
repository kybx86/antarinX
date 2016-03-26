classdef LinearRegressionModel
    
    properties
        
        X
        y
        num_iterations
        learning_rate
        bias = false
        optim_method = ''
        train_data_percentage  
        show_costFunction = false
        show_optimParams = false
        show_data = false
        show_fit = false
        normalize_features %not yet implemented
         
    end 
    
end