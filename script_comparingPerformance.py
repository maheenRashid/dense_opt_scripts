import util;
import os;
import visualize;
import scipy.misc;
import numpy as np;
def main():

	# dir_flo='/disk2/aprilExperiments/flo_subset_transfer';
	# dir_model_results='/disk2/aprilExperiments/flo_subset_predictions';
	# dir_meta_im='/disk2/marchExperiments/youtube';
	# util.mkdir(dir_model_results);

	# flo_files=[file_curr for file_curr in os.listdir(dir_flo) if file_curr.endswith('.flo')];
	# img_files_all=[];

	# for file_curr in flo_files:
	# 	video_name=file_curr[:file_curr.index('.')]
	# 	img_file=os.path.join(dir_meta_im,video_name,'images_transfer',file_curr.replace('.flo','.jpg'));
	# 	img_files_all.append(img_file+' 1');

	# util.writeFile(os.path.join(dir_model_results,'test.txt'),img_files_all);

	# # /disk2/aprilExperiments/flo_subset_predictions/test.txt /home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel 1

	# return


	dir_flo='/disk2/aprilExperiments/flo_subset_transfer';
	dir_meta_im='/disk2/marchExperiments/youtube';

	out_dir_flo_im='/disk2/aprilExperiments/flo_im';
	out_dir_tif_im='/disk2/aprilExperiments/tif_im';

	util.mkdir(out_dir_flo_im);	
	util.mkdir(out_dir_tif_im);

	out_file_html='/disk2/aprilExperiments/flo_im_visualize.html';
	rel_path_img=['/disk2','../../../..'];
	rel_path_tif=['/disk2','../../../..'];
	rel_path_flo=['/disk2','../../..'];

	flo_files=[os.path.join(dir_flo,file_curr) for file_curr in os.listdir(dir_flo) if file_curr.endswith('.flo')];
	# flo_files=flo_files[:10];
	

	img_paths_all=[];
	captions_all=[]
	for flo_file in flo_files:
		flo_just_name=flo_file[flo_file.rindex('/')+1:flo_file.rindex('.')];
		video_name=flo_just_name[:flo_just_name.index('.')];
		flo_file_np=os.path.join(out_dir_flo_im,flo_just_name+'.npy');
		if os.path.exists(flo_file_np):
			continue;
		print flo_file
		try:
			flo=util.readFlowFile(flo_file,flip=False)
		except:
			print 'ERROR';
			continue;

		
		np.save(flo_file_np,flo);

		print flo.shape
		out_flo_name_x=os.path.join(out_dir_flo_im,flo_just_name+'_x.png');
		visualize.saveMatAsImage(flo[:,:,0],out_flo_name_x);
		out_flo_name_y=os.path.join(out_dir_flo_im,flo_just_name+'_y.png');
		visualize.saveMatAsImage(flo[:,:,1],out_flo_name_y);

		jpg_name=os.path.join(dir_meta_im,video_name,'images_transfer',flo_just_name+'.jpg');
		tif_name=os.path.join(dir_meta_im,video_name,'images_transfer',flo_just_name+'.tif');

		tif=scipy.misc.imread(tif_name);
		print tif.shape,np.min(tif),np.max(tif);
		tif_just_name=flo_just_name;
		out_tif_name_x=os.path.join(out_dir_tif_im,tif_just_name+'_x.png');
		visualize.saveMatAsImage(tif[:,:,0],out_tif_name_x);
		out_tif_name_y=os.path.join(out_dir_tif_im,tif_just_name+'_y.png');
		visualize.saveMatAsImage(tif[:,:,1],out_tif_name_y);

		img_paths_all.append([jpg_name.replace(rel_path_img[0],rel_path_img[1]),out_flo_name_x.replace(rel_path_flo[0],rel_path_flo[1]),out_flo_name_y.replace(rel_path_flo[0],rel_path_flo[1]),
								out_tif_name_x.replace(rel_path_tif[0],rel_path_tif[1]),out_tif_name_y.replace(rel_path_tif[0],rel_path_tif[1])]);
		captions_all.append([flo_just_name,'x_flo','y_flo','cluster_x','cluster_y']);

	visualize.writeHTML(out_file_html,img_paths_all,captions_all,300,300);
	






if __name__=='__main__':
	main();