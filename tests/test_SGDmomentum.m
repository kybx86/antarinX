% test SGD momentum

N = 4;
D = 5;

w  = linspace(-0.4, 0.6, N*D); w  = reshape(w, [N, D]);
dw = linspace(-0.6, 0.4, N*D); dw = reshape(dw, [N, D]);
v  = linspace(0.6, 0.9, N*D);  v  = reshape(v, [N,D]);

config = struct('lr_rate', 0.001, 'velocity', v, 'momentum', 0.9);

next_w = MinMethods.sgd_momentum(w, dw, config);

% MATLAB reshapes in a different order than python does. FOr python to be
% equal to matlab,  we need order = 'F'. for fortran style.
expected_next_w =...
 [0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789;
 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526;
 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263;
 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096  ];


error = next_w - expected_next_w;
E = sum(sum(next_w)) - sum(sum(expected_next_w));

fprintf('\n Net Error %d', E);
if E < 1e-8
 fprintf('\n OK! SGD_Momentum test passed \n');
end


%e =  max(abs(x - y) / ( max( 1e-8, abs(x) + abs(y))))
