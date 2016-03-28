function [loss, grad]=nnn(nn_params, samples, labels, lambda, SET)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
input = SET.input ;
hidden1 = SET.hidden1;
output = SET.output;

%i want to just create a network with x inputs, x layers, x outputs. 
%the dificulity is that we have to deal with creating weights and reshaping
% but actually not that difficult 

weights1 = reshape(nn_params(1:hidden1 * (input + 1) ), ...
 [hidden1, (input + 1)] ); %25 401 

weights2 = reshape(nn_params((1 + (hidden1* (input + 1))):end), ...
    [ output, (hidden1 + 1) ] ); %10 26

   
[m, n] =size(samples) %5000 401

grad_weights1 = zeros(size(weights1)); %25 401
grad_weights2 = zeros(size(weights2)); %10 26 

loss = 0;

% the problem here is that labels cant be randomized or split
one_hot = ops.one_hot_matrix(labels,m, output);


%feedforward

% layer 1
a1 = samples; %5000 401 

% layer 2
z2 = a1 * weights1';
a2 = ops.sigmoid(z2);
a2 = ops.add_bias(a2);
size(a2)

%layer 3
z3 = a2 * weights2';
a3 = ops.sigmoid(z3);
hypothesis = a3; 


 




end