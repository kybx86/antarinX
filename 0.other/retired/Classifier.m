%%get rid of it


classdef Classifier
  properties
    X_tt
    y_tt
    weights
  end
  methods (Static = false)
    %the problem here is that we keep on copying the data around.
    %this should be the case! the data should exist in only one place and
    %get called from there. But then we cant create objects unless they
    %have the data on which to operate on. (or maybe they can, no
    %properties, just reference)
    
    function obj = Classifier(data_model)
      obj.X_tt = data_model.X_tt;
      obj.y_tt = data_model.y_tt;
    end
    
    function accuracy = evaluate(obj, weights)
      
      sample_size        = size(obj.X_tt,1);
      probability_vector = zeros(sample_size, 1);
      
      for i = 1:sample_size
        if ops.sigmoid(obj.X_tt(i,:)*weights) >= 0.5
          probability_vector(i) = 1;
        else
          probability_vector(i) = 0;
        end
      end
      
      accuracy = mean(double(probability_vector == obj.y_tt)) * 100;
      
      
    end
    
    function accuracy = multi_evaluate(obj, weights_tensor)
     
     m = size(X, 1);
     num_labels = size(all_theta, 1);

     p = zeros(size(X, 1), 1);
     
     
     X = [ones(m, 1) X];
     
     
     hypothesis = sigmoid(obj.X * all_theta' ); % returns n X k matrix with elements values between 0-1 depending on fit.
     [p_max, k_class] = max(hypothesis, [], 2);
     % p_max contains the greatest probability of the number for that sample x and k_class contains the class k
     %at which that p_max was found. Thus, the highest probability corresponds
     %to the highest class prediction for a single x.
     
     p = k_class;
     
    end
    
    
  end
end