classdef LinearRegModel
    properties
        X
        y
        params
        feature_normalize = false;
        alpha 
    end
    
    methods (Static = false)
        
        function model = LinearRegModel(input, target, feature_normalize) %
            %class constructor
            if(nargin > 0)
                
                model.X = input;
                model.y = target;
                model.params = zeros(size(input,2)+1, 1);
                model.feature_normalize = feature_normalize;
                
            end
            
        end
        
        
        function model = addBias(model)
            
            sample_size = size(model.X, 1);
            model.X = [ones(sample_size, 1) model.X];
            
        end
        
        function model = setAlpha(model, alpha)
           model.alpha = alpha; 
        end
            
        
        %         function [model, mu_mean, sigma_std] = normalizeFeatures(model)
        %
        %             mu_mean = zeros(1, size(model.x, 2))
        %
        %             sigma_std = zeros(1, size(model.x, 2))
        %
        %             mu_mean = mean(model.x)
        %
        %             normalized_features = model.x - mu_mean;
        %             std_sigma = std(normalized_features, 'dim', 1); %possible way to reference
        %             model.x = normalized_features / std_sigma ;
        %
        %         end
        
    end
    
end