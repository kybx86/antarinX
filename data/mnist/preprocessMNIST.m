clear; close all;
%changed to ops.
labels  = loadMNISTLabels('train-labels.idx1-ubyte');
samples = loadMNISTImages('train-images.idx3-ubyte');

samples = samples'; 

isZero = labels == 0; 
labels(isZero) = 10;  

save('fullMNIST.mat', 'samples', 'labels');

% % verify encoding is correct by visualizing image vector and comparing it
% % to label 

% for i = 1: 100
%  %sel = randperm(60000,1);
%  displayData(samples(i, :));
%  disp(labels(i))
%  pause
% end


