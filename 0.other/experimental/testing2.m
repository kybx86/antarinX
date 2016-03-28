layer = [400, 25, 10];

a1 = ones(1,3);
a2 = 2*ones(3,4);
a3 = 3*ones(2,2);


%q = [a1(:); a2(:); a3(:) ]

 %rep = 0;
 rep1 = ones(1,3); 
 rep1 = rep1(:); % first param 
 
for i = 1:2
 %rep1 = i*ones(i,3);
 rep2 = i*i*ones(i,2);
 %rep1 = rep1(:);
 rep2 = rep2(:);
 rep = [rep1 ; rep2];
end

BIAS = 1;

% weights1 = reshape(rep(1:hidden1 * (input + 1) ), ...
%  [hidden1, (input + 1)] ); %25 401 
% 
% weights2 = reshape(rep((1 + (hidden1* (input + 1))):end), ...
%     [ output, (hidden1 + 1) ] ); %10 26

   % 1 : (m*n+bias)
   weights1 = reshape( rep(1: layer(2)*(layer(1)+BIAS) ), ...
 [layer(2), (layer(1) + BIAS)] ); %25 401 


 weights2 = reshape( rep( 1+ layer(2)*(layer(1)+BIAS):...
  layer(3)*(layer(2)+BIAS)), [layer(3), (layer(2)+BIAS)])

weights2 = reshape(rep((1 + (hidden1* (input + 1))):end), ...
    [ output, (hidden1 + 1) ] ); %10 26