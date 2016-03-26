classdef LinearRegTrainer
    
    properties
        lr_model
        learning_rate
        num_iterations
        
        
    end
    
    methods (Static = false)
        
        function trainer = LinearRegTrainer(lr_model, learning_rate, num_iterations)
            
            trainer.lr_model       = lr_model;
            trainer.learning_rate  = learning_rate;
            trainer.num_iterations = num_iterations;
            
        end
        
        function trainer = gradient_descent(trainer)
            
            fprintf('Running Gradient Descent ...\n')
            
            % short hand notations
            sample_size    = length(trainer.lr_model.y);
            alpha          = trainer.learning_rate;
            num_iters      = trainer.num_iterations; % number of gradient steps on params
            cost_history   = zeros(num_iters,1);
            
            X     = trainer.lr_model.X;
            y     = trainer.lr_model.y;
            m     = sample_size;
            
            for iter = 1:num_iters
                
                hypothesis = X * trainer.lr_model.params;
                trainer.lr_model.params = trainer.lr_model.params - (alpha/m)* X'*(hypothesis-y);
                cost_history(iter) = LinearRegTrainer.compute_cost(X,y,trainer.lr_model.params); %static method call
                
            end
            
            fprintf('Theta found by gradient descent: \n');
            disp(trainer.lr_model.params);
            
            
            
        end
        
        
        function trainer = normal_equation(trainer) %avoid normalEq
           
            X     = trainer.lr_model.X;
            y     = trainer.lr_model.y;
            
            trainer.lr_model.params = (X'*X)\X'*y ;
            
            
            fprintf('Theta found by normal equation: \n');
            disp(trainer.lr_model.params);
            
            
        end
        
        
        
        function trainer = visualize_optimization(trainer)
            
            fprintf('Visualizing J(theta_0, theta_1) ...\n')
            X     = trainer.lr_model.X;
            y     = trainer.lr_model.y;
            theta = trainer.lr_model.params;

            
            theta0_vals = linspace(-10, 10, 100);
            theta1_vals = linspace(-1, 4, 100);
            
            % initialize J_vals to a matrix of 0's
            J_vals = zeros(length(theta0_vals), length(theta1_vals));
            
            % Fill out J_vals
            for i = 1:length(theta0_vals)
                for j = 1:length(theta1_vals)
                    t = [theta0_vals(i); theta1_vals(j)];
                    J_vals(i,j) = LinearRegTrainer.compute_cost(X, y, t);
                end
            end
            

            J_vals = J_vals';
            % Surface plot
            subplot(2,1,1)
            surf(theta0_vals, theta1_vals, J_vals)
            xlabel('\theta_0'); ylabel('\theta_1');
            
            % Contour plot
            % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
            subplot(2,1,2)
            contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
            xlabel('\theta_0'); ylabel('\theta_1');
            hold on;
            plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
            
        end
        
        
        
    end
    
    methods (Static = true)
        
        function cost = compute_cost(X, y, theta)
            
            sample_size = length(y);
            
            hypothesis = X * theta;
            cost = ((hypothesis-y)'*(hypothesis-y)) / (2*sample_size);
            
        end
        
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
end
