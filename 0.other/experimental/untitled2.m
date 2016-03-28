% reps=20;
% fprintf(['|',repmat('-',1,reps),'|']) %make the "background"
% fprintf(repmat('\b',1,reps+1)) %send the cursor back to the start
% 
% for ii=1:reps
%   fprintf('*')
%   pause(0.01)
% end
% 
% fprintf('\n')


%# Generate the data
% % Measurement1 = {[0.33 0.23 0.34 -32.32]; [-132.3 32.1 32.23 -320.32]};
% % Measurement2 = {433.2; 3.2};
% % TextStuff = {'The cat who ate the rat'; 'The dog who ate the cat'};
% % s = cell2struct([Measurement1, Measurement2, TextStuff], ...
% %     {'Measurement1', 'Measurement2', 'TextStuff'}, 2); 
% % 
% % str_format = @(tag, value)sprintf('%s:%s', tag, value);
% % 
% % # Iterate over the data and print it on the same figure
% % figure
% % for i = 1:length(s)
% % 
% %     # Clear the figure
% %     clf, set(gcf, 'color', 'white'), axis off
% % 
% %     # Output the data
% %     text(0, 1, str_format('Measurement1', num2str(s(i).Measurement1)));
% %     text(0, 0.9, str_format('Measurement2', num2str(s(i).Measurement2)));
% %     text(0, 0.8, str_format('TextStuff', s(i).TextStuff))
% % 
% %     # Wait until the uses press a key
% %     pause
% % end