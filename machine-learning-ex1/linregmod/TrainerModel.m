
classdef TrainerModel
    
    properties
        X_tr    % matrix training data set
        y_tr    % vector training data labels
        X_cv    % matrix cross-validation data set
        y_cv    % vector cross-validation data labels
        params  % vector theta parameters
        model   % object general LinearRegressionModel
        cost    % vector J_history 
    end
    
    methods (Static = false)
        
        %class constructor
        function trainer_model = TrainerModel(model) 
            
            sample_size = size(model.X, 1);
            sel = randperm(size(model.X, 1));
            train_sel = sel(1:round(model.train_data_percentage * sample_size));
            cv_sel = sel(length(train_sel)+1:end);
            
            
            %training set
            trainer_model.X_tr   = model.X(train_sel, :);
            trainer_model.y_tr   = model.y(train_sel, :);
            trainer_model.params = zeros(size(trainer_model.X_tr, 2)+1, 1);
            
            %cross-validation set
            trainer_model.X_cv   = model.X(cv_sel, :);
            trainer_model.y_cv   = model.y(cv_sel, :);
            
            trainer_model.model = model;

            % model.feature_normalize = feature_normalize;
            
        end
        
        function trainer_model = addBias(trainer_model)
            
            X_tr_sample_size = size(trainer_model.X_tr, 1);
            X_cv_sample_size = size(trainer_model.X_cv, 1);
            
            trainer_model.X_tr = [ones(X_tr_sample_size, 1) trainer_model.X_tr];
            trainer_model.X_cv = [ones(X_cv_sample_size, 1) trainer_model.X_cv];
            
        end
        
        function trainer_model = trainSGD(trainer_model)
            
            fprintf('Running Gradient Descent ...\n')
            
            % variable shorthand notation
            X_train           = trainer_model.X_tr;
            y_train           = trainer_model.y_tr;
            train_batch_size  = length(trainer_model.y_tr);
            alpha             = trainer_model.model.learning_rate;
            num_iters         = trainer_model.model.num_iterations; % number of gradient steps on params
            cost_history      = zeros(num_iters, 1);
            
            for iter = 1:num_iters
                hypothesis = X_train * trainer_model.params;
                trainer_model.params = trainer_model.params - (alpha/train_batch_size)* X_train'*(hypothesis-y_train);
                cost_history(iter) = TrainerModel.computeCost(X_train, y_train, trainer_model.params);
            end
            
            trainer_model.cost = cost_history;
            
            fprintf('Theta found by gradient descent: \n');
            disp(trainer_model.params);
            
            
            
        end
        
        function trainer_model = trainNormalEquation(trainer_model)
            
            X     = trainer_model.X_tr;
            y     = trainer_model.y_tr;
            
            trainer_model.params = (X'*X)\(X)'*y ;
            
            fprintf('Theta found by normal equation: \n');
            disp(trainer_model.params);
            
        end
        
        function showTrainingCost(trainer_model)
            fprintf('Visualizing J(iterations) ...\n')
            plot(trainer_model.cost);
            hold on;
        end
        
        function showOptimParams(trainer_model)
                        
            X     = trainer_model.X_tr;
            y     = trainer_model.y_tr;
            theta = trainer_model.params;
            
            if size(theta,1) > 2
                fprintf('ERROR: (showOptimParams) cannot visualize more than 2 parameters \n')
                return
            else
               fprintf('Visualizing J(params) ...\n')
               
                theta0_vals = linspace(-10, 10, 100);
                theta1_vals = linspace(-1, 4, 100);
                
                % initialize J_vals to a matrix of 0's
                J_vals = zeros(length(theta0_vals), length(theta1_vals));
                
                % Fill out J_vals
                for i = 1:length(theta0_vals)
                    for j = 1:length(theta1_vals)
                        t = [theta0_vals(i); theta1_vals(j)];
                        J_vals(i,j) = TrainerModel.computeCost(X, y, t);
                    end
                end
                
                
                J_vals = J_vals';
                % Surface plot
                figure
                subplot(2,1,1)
                surf(theta0_vals, theta1_vals, J_vals)
                xlabel('\theta_0'); ylabel('\theta_1');
                
                % Contour plot
                % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
                subplot(2,1,2)
                contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
                xlabel('\theta_0'); ylabel('\theta_1');
                hold on;
                subplot(2,1,2)
                plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
                
            end
            
            
        end
        
        function showTrainingData(trainer_model)

            if size(trainer_model.params, 1) > 2
                fprintf('ERROR: (showTrainingData) cannot visualize more than 2 parameters \n')
                return
            else
                figure
                plot(trainer_model.X_tr(:,2:end), trainer_model.y_tr, 'rx', 'MarkerSize', 10)
            end
            
        end
        
        function showFit(trainer_model)
            
            if size(trainer_model.params, 1) > 2
                fprintf('ERROR: (showFit) cannot visualize more than 2 parameters \n')
                return
            else
                % Plot the linear fit
                hold on; % keep previous plot visible
                plot(trainer_model.X_tr(:,2:end),trainer_model.y_tr, 'r+')
                plot(trainer_model.X_tr(:,2:end), trainer_model.X_tr*trainer_model.params, '-') %superimposing fit on data
                legend('Training data', 'Linear regression')
                hold off % don't overlay any more plots on this figure
            end
            
        end
        
    end
    
    
    methods (Static = true)
        
        function cost = computeCost(X, y, theta)
            
            m = length(y);
            hypothesis = X * theta;
            cost = ((hypothesis-y)' * (hypothesis-y)) / (2*m);
            
        end
        
        
    end
    
    
    
end %%end of class