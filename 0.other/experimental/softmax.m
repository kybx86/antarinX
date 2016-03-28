clear; close all;

% x = randperm(50);
% x = reshape(x, [10, 5])
%
x = [ 0.2735878   0.09426133 -0.06251958 -0.12569355  0.06207533;
 -0.0150762   0.01536743 -0.06184854 -0.02382389  0.18927352;
 0.02090436 -0.19201504  0.03434708 -0.0358352  -0.07544599;
 -0.029005   -0.04944675 -0.09731952 -0.15163395 -0.07069614;
 -0.03438096 -0.03570672 -0.0235415   0.10540935  0.0226413 ;
 -0.02840648  0.16691355 -0.10655996 -0.00175877 -0.11682774;
 -0.22835654 -0.03403389 -0.14477884 -0.11684813  0.05308806;
 0.02528177 -0.2041859  -0.04119211  0.03081665  0.06732221];

m = 8;
n = 5;
x = reshape(x, [m, n]);
%x = 0.1*randn(m, n);
%y = randi([1, n], 1, m); %x = reshape(x, [10, 5]);
y = [4 4 1 1 3 2 4 4];
y = y';
Y = ops.dense_to_hot(y);

[softmax, dloss] = Objectives.softmax('Hypothesis', x, 'Target', Y, 'compute_dx', false);



[~, d] = max(Y, [],2); %exctracts 1 from matrix


x_max = max(x, [], 2);
x_max = repmat(x_max, [1, n]);

e_x = exp(x - x_max); %probs = np.exp(x - np.max(x, axis=1, keepdims=True))

den = sum(e_x, 2);
den = repmat(den, [1, n]);
e_x = e_x ./ den;  %probs /= np.sum(probs, axis=1, keepdims=True)
%check!

N = size(x, 1);

% --one way to go about it
% p = e_x(:, y)
% k = diag(p) %class match values

% --the BETTER way to go about it
idx = sub2ind(size(e_x), 1:N, d');
k   = e_x(idx); % either transpose y into column or 1:N into colum with reshape
e_x(idx) = e_x(idx) - 1; %operate directly on elements 

% -- the other way to go about it 
% flat = e_x(:);
% h = flat(idx) ;
% h = h + 100;
% e_x(idx) = h


% j = [ 0.20080661  0.23576243  0.17289115  0.20594173  0.22038515  0.18194767 0.23065626  0.21822726];
%
scores = - sum(log(k)) / N;


% for i = 1 : m
%  v(i) = e_x(i, y(i));
% end
% v


%[loss, dloss] = Objectives.softmax('H', x,'hotY', Y)

