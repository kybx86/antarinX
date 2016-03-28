% testing Affine layer


% forward test

%not a good way to test. because 1. we dont know what we are trying to
%test. and 2. (most importantly) we 'dont know' the correct answer.

num_inputs = 2;
input_shape = [4, 5, 6];
output_dim = 3;

input_size  = num_inputs * prod(input_shape);
weight_size = output_dim * prod(input_shape);

x = linspace(-0.1, 0.5, input_size);
X = reshape(x, [num_inputs, input_shape]);
w = linspace(-0.2, 0.3, weight_size);
W = reshape(w, [prod(input_shape), output_dim]);
b = linspace(-0.3, 0.1, output_dim);


row_dim = size(X, 1);
col_dim = prod(input_shape);
X_reshape = reshape(X, [row_dim, col_dim]);

[fout, ~ ] = Affine.forward('Input', X_reshape, 'Weights', W', 'AddBias', false);

b = repmat(b, [2,1]);
out = fout + b;

err = [ 1.49834967,  1.70660132,  1.91485297; 3.25553199,  3.5141327,   3.77273342]
E = out- err;

% not good