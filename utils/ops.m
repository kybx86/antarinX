classdef ops
 
 methods (Static = true)
  
  %% --DATA IMAGE UTILITIES
  
  function images = loadMNISTImages(filename)
   %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
   %the raw MNIST images
   
   fp = fopen(filename, 'rb');
   assert(fp ~= -1, ['Could not open ', filename, '']);
   
   magic = fread(fp, 1, 'int32', 0, 'ieee-be');
   assert(magic == 2051, ['Bad magic number in ', filename, '']);
   
   numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
   numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
   numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
   
   images = fread(fp, inf, 'unsigned char');
   images = reshape(images, numCols, numRows, numImages);
   images = permute(images,[2 1 3]);
   
   fclose(fp);
   
   % Reshape to #pixels x #examples
   images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
   % Convert to double and rescale to [0,1]
   images = double(images) / 255;
   
  end
  
  function labels = loadMNISTLabels(filename)
   %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
   %the labels for the MNIST images
   
   fp = fopen(filename, 'rb');
   assert(fp ~= -1, ['Could not open ', filename, '']);
   
   magic = fread(fp, 1, 'int32', 0, 'ieee-be');
   assert(magic == 2049, ['Bad magic number in ', filename, '']);
   
   numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
   
   labels = fread(fp, inf, 'unsigned char');
   
   assert(size(labels,1) == numLabels, 'Mismatch in label count');
   
   fclose(fp);
   
  end
  
  function  display_CIFAR_image()
   % need to load data
   for i = 1:200
    sel = randperm(10000, 1);
    x = data(sel, :, :);
    xr = reshape(x, [], 64);
    %xr = reshape(x, [32, 32, 3]);
    imagesc(xr)
    pause
   end
   
  end
  
  function [h, display_array] = display_MNIST(X, example_width)
   %DISPLAYDATA Display 2D data in a nice grid
   %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
   %   stored in X in a nice grid. It returns the figure handle h and the
   %   displayed array if requested.
   
   % Set example_width automatically if not passed in
   if ~exist('example_width', 'var') || isempty(example_width)
    example_width = round(sqrt(size(X, 2)));
   end
   
   % Gray Image
   colormap(winter);
   
   % Compute rows, cols
   [m n] = size(X);
   example_height = (n / example_width);
   
   % Compute number of items to display
   display_rows = floor(sqrt(m));
   display_cols = ceil(m / display_rows);
   
   % Between images padding
   pad = 4;
   
   % Setup blank display
   display_array = - ones(pad + display_rows * (example_height + pad), ...
    pad + display_cols * (example_width + pad));
   
   % Copy each example into a patch on the display array
   curr_ex = 1;
   for j = 1:display_rows
    for i = 1:display_cols
     if curr_ex > m,
      break;
     end
     % Copy the patch
     
     % Get the max value of the patch
     max_val = max(abs(X(curr_ex, :)));
     display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
      pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
      reshape(X(curr_ex, :), example_height, example_width) / max_val;
     curr_ex = curr_ex + 1;
    end
    if curr_ex > m,
     break;
    end
   end
   
   
   % Display Image
   
   %Fig = figure(1);
   %set(Fig,                        ...
   %         'NumberTitle', 'off',         ...
   %         'Name',         mfilename,    ...
   %         'Color',        [1 1 1] ); %change background color here [R G B]
   
   h = surf(display_array, [-1 1]);
   
   % Do not show axis
   axis image off
   
   drawnow;
   
  end
  
  function image_to_binary(image)
   
   image = imread(image);
   bw=im2bw(image,colormap,0.28);
   imshow(bw);
   %ask user to write the name of the file. to save.
   %save('image.mat','bw');
   
  end
  
  
  
  %% --CONSTANTS, SEQUENCES, RANDOM VALUES
  
  function tensor = new_1D_tensor(size)
   %-- Creates a column tensor R^n.
   % Args:
   %    tensor: tensor, double - R^
   % Returns:
   %    tensor: R^n weight tensor for samples
   
   tensor = zeros(size, 1);
   
  end
  
  function tensor = new_n_tensor(num_labels, dimension)
   % -- Creates zeros new matrix
   tensor = zeros(num_labels, dimension);
  end
  
  function [samples, labels] = load_data(nameOfFile)
   % -- loads numeric samples and labels from file
   % -- the file must adhere to:
   %           samples = (all_rows, column1 to last_column-1)
   %           labels  = (all_rows, last_colum)
   
   
   all_data = load(nameOfFile);
   samples = all_data(:, 1:end-1);
   labels = all_data(:, end);
   
  end
  
  function tensor = unroll_tensors(tensor1, tensor2)
   % -- Unrolls two tensors into flattened linear vector
   tensor = [tensor1(:) ; tensor2(:)];
  end
  
  function num_labels = count_labels(tensor)
   % -- Counts labels in a 1D array
   % Args:
   %    tensor: tensor, double - [m, 1]
   % Returns:
   %    num_labels: number of labels in input tensor
   
   tensor = sort(tensor);
   sample_size = size(tensor);
   label_count = 1;
   
   for m = 1:sample_size - 1
    lagging_tracker = tensor(m);
    leading_tracker = tensor(m+1);
    if(lagging_tracker == leading_tracker)
     continue
    else
     label_count = label_count + 1;
    end
   end
   num_labels = label_count;
  end
  
  function tensor = rand_init_weights(~, L_in, ~, L_out)
   % This method is implemented in NN_utils.m
   % --Random initialization of layer a weights
   % Args:
   %    L_in, int,  number of incoming connections
   %    L_out, int, number of outgoing connections
   % Returns:
   %    NetworkWeight object with initialized 'layer_weights'
   %
   % --Note: The first row of W corresponds to the bias units
   
   epsilon = sqrt(6) / sqrt(L_in + L_out);
   
   BIAS = 1;
   
   layer_weights = rand(L_out, BIAS + L_in) * (2*epsilon) - epsilon;
   
   W = zeros(L_out, 1 + L_in);
   
   tensor = W;
   
  end
  
  
  %% --MATH
  
  function tensor = sigmoid(tensor)
   % --implemented in Transfers.m
   
   tensor = 1 ./ (1 + exp(-tensor));
   
  end
  
  function g = sigmoidGradient(z)
   % -- implemented in Transfers.m
   % -- sigmoidGradient returns the gradient of the sigmoid function
   % -- evaluated at z
   %   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
   %   evaluated at z. This should work regardless if z is a matrix or a
   %   vector.
   g = zeros(size(z));
   
   % derivative of sigmoid
   g = ops.sigmoid(z).*(1-ops.sigmoid(z));
   
   
  end
  
  
  %% --TRANSFORMAIONS & ENCODINGS
  
  function tensor = add_bias(tensor)
   %  we should modify it to add a bias value of our choosing: i.e...
   %  add_bias(tensor, bias_value): bias = ones(size(tensor,1 )) * bias_value
   %                                tensor = [bias, tensor]
   %
   % --Adds column of ones to a 2D 'tensor'.
   % Args:
   %    tensor: tensor, double - R^mXn
   % Returns:
   %    tensor: concatenated bias to tensor
   
   tensor =  [ones(size(tensor, 1), 1), tensor] ;
   
  end
  
  function tensor = remove_bias(tensor)
   % --Removes first bias column of matrix
   tensor = [tensor(:, 2:end)];
  end
  
  function tensor = omit_replace_bias(tensor)
   % --Replaces bias ones column for zeros column for regularization semantic
   tensor = [zeros(size(tensor, 1), 1), tensor(:, 2:end)];
   
  end
  
  function data_model = split_data(~, samples,...
    ~, labels,...
    ~, one_hot_tensor,...
    ~, randomSel,...
    ~, SET)
   % --splits data into training, cross_validation and testing sets.
   % Args:
   %    tensor_X: samples, [m+bias, n] or [m, n]
   %    tensor_y: labels,  [m, 1]
   %    tensor_Y, one_hot_tensor, [m, k]
   %    randomSel, bool, specifiy for randomized selection or linear
   %
   % Returns:
   %    X_tr: selected samples for training
   %    y_tr: paired labels of X_tr (in one hot)
   %    yraw_tr: paired labels (in vector form)
   %    X_cv: selected samples for cross validation
   %    y_cv: paired labels of X_cv (in one hot)
   %    yraw_cv: paired label (in vectors form)
   %    X_tt: selected samples for testing
   %    y_tt: paired labels of X_tt (in one hot)
   %    yraw_tt: paired labels (in vectors form)
   %
   
   [sample_size, ~] = size(samples);
   
   assert(SET.train_split + SET.val_split + SET.test_split == 100, ...
    'ERROR: Check Data Splits')
   
   train_split = SET.train_split / 100;
   val_split   = SET.val_split   / 100;
   test_split  = SET.test_split  / 100;
   
   
   if randomSel == true
    sel = randperm(sample_size);
   else
    sel = [1 : sample_size];
   end
   
   train_sel = sel(1:round(train_split*sample_size));
   val_sel   = sel(length(train_sel)+1:round(val_split*...
    sample_size)+length(train_sel));
   test_sel  = sel(length(val_sel)+length(train_sel)+1:end);
   
   %training set
   X_tr    = samples(train_sel, :);
   y_tr    = one_hot_tensor(train_sel, :);
   yraw_tr = labels(train_sel, :);
   
   %validation set
   X_cv    = samples(val_sel, :);
   y_cv    = one_hot_tensor(val_sel, :);
   yraw_cv = labels(val_sel, :);
   
   %test set
   X_tt    = samples(test_sel, :);
   y_tt    = one_hot_tensor(test_sel, :);
   yraw_tt = labels(test_sel, :);
   
   data_model = ...
    DataModel(X_tr, y_tr, X_cv, y_cv, X_tt, y_tt, yraw_tr, yraw_cv, yraw_tt);
   
  end
  
  function tensor = one_hot_matrix(labels, num_labels)
   % --the data labels do not have to be sorted
   % --this function breaks down when labels include a value of 0.
   % --for that reason, it is better to use the 'dense_to_hot' below.
   % Args:
   %    labels, vector of labels containing k class unique values.
   %          But MUST NOT contain zero value as a class.
   %    num_labels, scalar, number of unique classes in labels.
   %
   % Returns:
   %     one_hot_matrix (m, K_classes)
   
   sample_size = size(labels, 1);
   id_matrix = eye(num_labels);
   one_hot_matrix = zeros(sample_size, num_labels);
   
   for i = 1:sample_size
    one_hot_matrix(i, :) = id_matrix(labels(i), :);
   end
   
   tensor = one_hot_matrix;
   
  end
  
  function tensor = dense_to_hot(labels)
   % --dense_to_hot is more robust in encoding vectors into identity
   % --matrices. Additionally, it also uses a clever method to count the
   % --number of classes
   % Args:
   %    labels, vector of labels containing k class unique values.
   %
   % Returns:
   %     one_hot_matrix (m, K_classes)
   
   value_labels = unique(labels);
   num_labels = length(value_labels);
   % --NOTE: if we ever need num_labels we can extract it by the 2nd
   % -- dimension of the one_hot encoding
   
   sample_size = size(labels, 1);
   one_hot_tensor = zeros(sample_size, num_labels);
   
   for i = 1:num_labels
    one_hot_tensor(:,i) = (labels == value_labels(i));
   end
   
   tensor = one_hot_tensor;
   
  end
  
  function tensor = Z_norm(tensor)
   % --Z_norm pre processes the input data by STANDARIZING the features with
   % --a normal distribution of mean=0 and std=1.  z = (x - mean)/std.
   % --NOTE: this is different from normaliztion (min_max_scaling) since the
   % --inputs are not forced to 0-1. (and Z_norm) always works much better.
   % --IMPORTANT: the mean and the std must be stored and later used to
   % --normalize the test set and calculate the accuracy and further
   % --predictions
   %
   % Args:
   %    MxN numeric tensor
   % Returns:
   %     MxN numeric tensor of z-score standarized feartures
   
   mu = mean(tensor, 1);
   mu_mat = repmat(mu, [size(tensor, 1), 1]);
   norm  = tensor - mu_mat;
   sigma = std(norm);
   sigma_mat = repmat(sigma, [size(tensor, 1), 1]);
   z = norm./sigma_mat;
   
   tensor = z;
  end
  
  
  %% --REGRESSION
  
  function [loss, grad] = multi_logreg_loss(X, y, lambda)
   %  needs to be adapted again to fit new api.
   % -- multi_logreg_loss is a function to calculate a multi-class logistic
   % -- regression model.  It uses sigmoid activation functions and the
   % -- cross_entropy loss function, and incorporates L2 regularization.
   % -- This also computes the gradient of the multi_logreg_loss.
   %
   % --Note: this method was written to feed into a @fminunc 2nd order optimizer,
   %         hence why the grad is flattened.
   % Args:
   %     X, samples, matrix -- ALL data must be a single batch. !
   %     y, labels, vector -- ALL labels must be a single batch. !
   %
   % Returns:
   %     loss, scalar, value of cross entropy loss
   %     grad, flattened grad vector.
   
   
   loss = @multi_logreg_loss; % abstract function for specific use in graph
   
   function [loss, grad] = multi_logreg_loss(theta, y) %added y
    loss = 0;
    grad = 0;
    
    m = length(y);
    hypothesis = ops.sigmoid(X * theta);
    reg_theta  = [0; theta(2:end, :)]; %no bias reg
    log_loss   = (1/m) * (-y'*log(hypothesis) - (1-y)'*log(1-hypothesis));
    reg_factor = (lambda / (2*m)) * (reg_theta' * reg_theta);
    loss       = log_loss + reg_factor;
    
    grad = (1/m)*(X'*(hypothesis - y)) + (lambda/m)*reg_theta;
    grad = grad(:);
   end
   
  end
  
  
  %% --EVALUATION
  
  
  function accuracy = calculate_accuracy(~, hypothesis, ~, target, ~, verbose)
   % -- Calculates accuracy against Target
   %
   % Args:
   %     hypothesis, (m x k )
   %     target, vector of targets (m, 1)
   %     verbose, boolean, verbose setting to display accuracy
   %
   % Returns:
   %       accurracy, scalar, percentage of max hypothesis matching target
   
   [~, pred_idx] = max(hypothesis, [], 2); % picks out max col_indices
   accuracy = ( mean( pred_idx == target) * 100);
   
   if verbose
    fprintf('\n Testing accuracy: %f \n', accuracy);
   end
   
  end
  
  function accuracy = multi_evaluate(samples, labels, weights)
   % this function does not adapt the new api.
   % -- Calculates accuracy against labels by calculating hypothesis
   % -- used in log reg.
   
   [sample_size, dim] = size(samples); %5000 401
   [num_labels, ~] = size(weights); % 10 401
   p = ops.new_1D_tensor(sample_size); %5000 1
   
   hypothesis = ops.sigmoid(samples * weights' ); %5000 10
   [p_max, k_class] = max(hypothesis, [], 2);
   %returns index with the largest value across a dimension of a tensor
   p = k_class;
   
   accuracy =  mean(double(p == labels)) * 100;
   
  end
  
  function accuracy = binary_evaluate(samples, labels, weights)
   % this function does not adapt the new api.
   % -- Calculates accuracy against labels by calculating hypothesis
   % -- used in log reg but.
   
   
   [sample_size, dim] = size(samples); %5000 401
   [num_labels, ~] = size(weights); % 10 401
   p = ops.new_1D_tensor(sample_size); %5000 1
   
   hypothesis = ops.sigmoid(samples * weights' ); %5000 10
   [p_max, k_class] = max(hypothesis, [], 2);
   
   p = k_class;
   probability = (round(p_max));
   
   accuracy =  mean(double(probability == labels)) * 100;
   
  end
  
  
  %% --SUMMARY
  
  function display_model_summary(SET, samples, num_labels, accuracy, weights)
   % this function does not adapt the new api.
   % -- Prints out summary of model
   
   fprintf('\n PROCEDURE COMPLETE \n');
   
   fprintf('\n Raw Data Summary  \n');
   fprintf('    Total Num of samples  : %d  \n', size(samples, 1));
   fprintf('    Total Num of features : %d  \n', size(samples, 2));
   fprintf('    Total Num of classes : %d  \n', num_labels);
   
   fprintf('\n Data Partition \n')
   fprintf('    Training  : %d percent \n', SET.train_split);
   fprintf('    Validation: %d percent \n', SET.val_split);
   fprintf('    Testing   : %d percent \n', SET.test_split);
   
   fprintf('\n Model Accuracy:  %f \n', accuracy);
   
   fprintf('\n Learned Parameters  \n')
   fprintf('    Contains %d classes with %d features each\n',...
    size(weights, 1), size(weights, 2));
   fprintf('\n Saving Learned Parameters ... "weights.mat" \n');
   
   save('weights.mat', 'weights');
   fprintf('\n')
   %toc %end timer
   fprintf('\n')
   fprintf('\n END OF SUMMARY \n');
   
  end
  
  
  
  
 end
 
end
