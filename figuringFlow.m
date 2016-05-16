% function [] = loadResults()
      
 %    rootFile = ['./theResult'];
 %    load(['./clusters.mat'], 'C');

 %    theFiles = dir([rootFile '/*.txt'])

 %    for i = 1:length(theFiles)
 
 %    	theNumFile = theFiles(i).name;

 %    	[~,theName, ~] = fileparts(theNumFile);

	% theImage = (fileread([rootFile '/' theNumFile]));
 %        theImage = imread(theImage(1:(end - 1)));
	

	file_name=['/disk2/aprilExperiments/deep_proposals/flow/results/44.h5']        
	h5disp(file_name, '/Outputs');
        theTemp = h5read(file_name, '/Outputs');
	theOutput = [];
	    for j = 1:size(theTemp,2)
		for k = 1:size(theTemp,1)
		     for l = 1:size(theTemp,3)
			theOutput = [theOutput theTemp(k,j,l)];
		     end
		end
            end
        
        N = assignToFlowSoft(theOutput(:), C);

	N = imresize(N, [size(theImage,1) size(theImage,2)]);
	out_file=['/disk2/temp/checking_h5.mat']
	save(out_file,'N');

% 	figure;
% 	makeArrowFlowFigure(N, 1, 20, 2, 0.5, 0.1, 0.2, theImage)
% end

