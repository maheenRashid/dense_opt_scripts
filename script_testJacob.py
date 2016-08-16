import os;
import util;
import random;
import visualize
import sys;

def dump_main():

	return

	# path_to_im='/disk2/aprilExperiments/deep_proposals/flow/results_images'
	# visualize.writeHTMLForFolder(path_to_im,ext='jpg',height=300,width=300)
	

	return

	dir_meta='/disk2/marchExperiments/finetuning_youtube_hmdb/check';
	dir_im_before=os.path.join(dir_meta,'results_images_before');
	dir_im=os.path.join(dir_meta,'results_images_llr_25000')
	test_file=os.path.join(dir_meta,'test.txt');

	out_file_html=os.path.join(dir_meta,'visualize.html');

	lines=util.readLinesFromFile(test_file);
	lines=[line_curr[:line_curr.index(' ')] for line_curr in lines];

	rel_path=['/disk2','../../../..'];
	img_paths=[];
	captions=[];
	for idx_line,im_file in enumerate(lines):
		img_org=im_file.replace(rel_path[0],rel_path[1]);
		img_before=os.path.join(dir_im_before,str(idx_line+1)+'.jpg').replace(rel_path[0],rel_path[1]);
		img_after=os.path.join(dir_im,str(idx_line+1)+'.jpg').replace(rel_path[0],rel_path[1]);
		img_paths.append([img_before,img_after]);
		captions.append(['original_model','finetuned_model']);

	visualize.writeHTML(out_file_html,img_paths,captions,height=300,width=300);


	


	return
	# '/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat'
	out_dir_train='/disk2/marchExperiments/finetuning_youtube_hmdb_llr'
	path_to_binary='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test';
	path_to_binary=os.path.join(path_to_binary,'build/examples/opticalflow/test_net_flow_bin.bin');
	# model_file=os.path.join(out_dir_train,'OptFlow_youtube_hmdb_iter_255000.caffemodel');

	path_to_matlab='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow'
	matlab_file='loadResults_script';
	matlab_file_path_template=os.path.join(path_to_matlab,'loadResults_script_template.m');
	matlab_file_path=os.path.join(path_to_matlab,'loadResults_script.m');

	mean_file=os.path.join(path_to_matlab,'opt_train_db.binaryproto');
	template_file=os.path.join(path_to_matlab,'opt_test_coarse_xavier_template.prototxt');
	clusters_file=os.path.join(path_to_matlab,'clusters.mat');
	# model_file=os.path.join(path_to_matlab,'final.caffemodel');
	model_file=os.path.join(out_dir_train,'OptFlow_youtube_hmdb_iter_25000.caffemodel');

	# out_dir='/disk2/aprilExperiments/deep_proposals/flow'
	out_dir='/disk2/marchExperiments/finetuning_youtube_hmdb/check';
	out_dir_results=os.path.join(out_dir,'results_llr_25000');
	# out_dir_results_before=os.path.join(out_dir,'results_before');
	# out_dir_results=os.path.join(path_to_matlab,'theResult');
	out_dir_images=os.path.join(out_dir,'results_images_llr_25000');
	# out_dir_images_before=os.path.join(out_dir,'results_images_before');

	util.mkdir(out_dir_images);
	# util.mkdir(out_dir_images_before);
	util.mkdir(out_dir_results);
	# util.mkdir(out_dir_results_before);

	test_file=os.path.join(out_dir,'test.txt');
	out_file_opt=os.path.join(out_dir,'opt_test_coarse_xavier.prototxt');

	# rel_path=['/disk2',rel_path_calc];

	# replace the text in opt_test_coarse and save it in out_dir
	with open(template_file,'rb') as f:
		opt_data=f.read();
	opt_data=opt_data.replace('$TEST_FILE',test_file);
	opt_data=opt_data.replace('$MEAN_FILE',mean_file);
	with open(out_file_opt,'wb') as f:
		f.write(opt_data);

	# change the test.sh command appropriately
	command_sh=[path_to_binary,out_file_opt,model_file,test_file,out_dir_results,'40 20 1'];
	command_sh=' '.join(command_sh);
	print command_sh
	print '____'
	# run the matlab command
	matlab_pre='matlab -nodisplay -nodesktop -r "cd ';
	with open(matlab_file_path_template,'rb') as f:
		mat_data=f.read();
	mat_data=mat_data.replace('$ROOT_FILE',"'"+out_dir_results+"'");
	mat_data=mat_data.replace('$CLUSTERS_FILE',"'"+clusters_file+"'");
	mat_data=mat_data.replace('$OUT_DIR',"'"+out_dir_images+"'");
	with open(matlab_file_path,'wb') as f:
		f.write(mat_data);

	# matlab_command=[matlab_pre+path_to_matlab,'rootFile="'+out_dir_results+'"','out_dir="'+out_dir_images+'"','clusters_file="'+clusters_file+'"',matlab_file,'exit','"'];
	
	matlab_command=[matlab_pre+path_to_matlab,matlab_file,'exit','"'];
	# print out_dir_results_before
	# print out_dir_images_before
	
	matlab_command=';'.join(matlab_command);
	print '____'
	print matlab_command

	# /group/leegrp/maheen_data/jacob/opticalflow; warpGetOpFlowPan;exit" >> "logs/"$HOST"_"$SLURM_ARRAY_TASK_ID.log
	# make html

	return

	num_to_test=300;
	# write the test.txt
	train_txt='/disk2/marchExperiments/finetuning_youtube_hmdb/train.txt'
	test_txt='/disk2/marchExperiments/finetuning_youtube_hmdb/check/test.txt'
	lines=util.readLinesFromFile(train_txt);
	random.shuffle(lines);
	lines=lines[:num_to_test];
	lines=[line_curr[:line_curr.index(' ')]+' 1' for line_curr in lines];
	util.writeFile(test_txt,lines);

	return

def getCommandForTest(test_file,model_file,gpu,batch_size=100,train_val_file=None):
	gpu=str(gpu);
	
	path_to_binary='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test';
	path_to_binary=os.path.join(path_to_binary,'build/examples/opticalflow/test_net_flow_bin.bin');

	path_to_matlab='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow'
	matlab_file='loadResults_script';
	matlab_file_path_template=os.path.join(path_to_matlab,'loadResults_script_template.m');
	matlab_file_path=os.path.join(path_to_matlab,'loadResults_script.m');

	mean_file=os.path.join(path_to_matlab,'opt_train_db.binaryproto');
	if train_val_file is None:
		template_file=os.path.join(path_to_matlab,'opt_test_coarse_xavier_template.prototxt');
	else:
		template_file=train_val_file;


	out_dir=test_file[:test_file.rindex('/')];
	out_dir_results=os.path.join(out_dir,'results');
	out_dir_images=os.path.join(out_dir,'results_images');
	
	util.mkdir(out_dir_results);
	util.mkdir(out_dir_images);


	out_file_opt=os.path.join(out_dir,'opt_test_coarse_xavier.prototxt');
	# print template_file
	# replace the text in opt_test_coarse and save it in out_dir
	with open(template_file,'rb') as f:
		opt_data=f.read();
	opt_data=opt_data.replace('$TEST_FILE',test_file);
	opt_data=opt_data.replace('$MEAN_FILE',mean_file);
	opt_data=opt_data.replace('$BATCH_SIZE',str(batch_size));
	with open(out_file_opt,'wb') as f:
		f.write(opt_data);

	# change the test.sh command appropriately
	command_sh=[path_to_binary,out_file_opt,model_file,test_file,out_dir_results,'40 20 '+gpu];
	command_sh=' '.join(command_sh);
	return command_sh	


def main(argv):

	test_file=argv[1];
	model_file=argv[2];
	gpu=argv[3];
	batch_size=100;
	
	path_to_binary='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test';
	path_to_binary=os.path.join(path_to_binary,'build/examples/opticalflow/test_net_flow_bin.bin');

	path_to_matlab='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow'
	matlab_file='loadResults_script';
	matlab_file_path_template=os.path.join(path_to_matlab,'loadResults_script_template.m');
	matlab_file_path=os.path.join(path_to_matlab,'loadResults_script.m');

	mean_file=os.path.join(path_to_matlab,'opt_train_db.binaryproto');
	# mean_file=os.path.join(path_to_matlab,'standard.binaryproto');
	template_file=os.path.join(path_to_matlab,'opt_test_coarse_xavier_template.prototxt');
	clusters_file=os.path.join(path_to_matlab,'clusters.mat');
	

	out_dir=test_file[:test_file.rindex('/')];
	out_dir_results=os.path.join(out_dir,'results');
	out_dir_images=os.path.join(out_dir,'results_images');
	
	util.mkdir(out_dir_results);
	util.mkdir(out_dir_images);


	out_file_opt=os.path.join(out_dir,'opt_test_coarse_xavier.prototxt');

	# replace the text in opt_test_coarse and save it in out_dir
	with open(template_file,'rb') as f:
		opt_data=f.read();
	opt_data=opt_data.replace('$TEST_FILE',test_file);
	opt_data=opt_data.replace('$MEAN_FILE',mean_file);
	opt_data=opt_data.replace('$BATCH_SIZE',str(batch_size));
	with open(out_file_opt,'wb') as f:
		f.write(opt_data);

	# change the test.sh command appropriately
	command_sh=[path_to_binary,out_file_opt,model_file,test_file,out_dir_results,'40 20 '+gpu];
	command_sh=' '.join(command_sh);
	print command_sh
	print '____'
	# run the matlab command
	matlab_pre='matlab -nodisplay -nodesktop -r "cd ';
	with open(matlab_file_path_template,'rb') as f:
		mat_data=f.read();
	mat_data=mat_data.replace('$ROOT_FILE',"'"+out_dir_results+"'");
	mat_data=mat_data.replace('$CLUSTERS_FILE',"'"+clusters_file+"'");
	mat_data=mat_data.replace('$OUT_DIR',"'"+out_dir_images+"'");
	with open(matlab_file_path,'wb') as f:
		f.write(mat_data);

	# matlab_command=[matlab_pre+path_to_matlab,'rootFile="'+out_dir_results+'"','out_dir="'+out_dir_images+'"','clusters_file="'+clusters_file+'"',matlab_file,'exit','"'];
	
	matlab_command=[matlab_pre+path_to_matlab,matlab_file,'exit','"'];
	# print out_dir_results_before
	# print out_dir_images_before
	
	matlab_command=';'.join(matlab_command);
	print '____'
	print matlab_command




if __name__=='__main__':
	main(sys.argv);

