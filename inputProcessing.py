import numpy as np;
import cv2;
import scipy.misc;
import os;
import util;
import multiprocessing;

def rescaleImAndSave((img_path,out_path,scale_factor,idx)):
	print idx;
	im=scipy.misc.imread(img_path);
	im_new=scipy.misc.imresize(im,scale_factor);
	scipy.misc.imsave(out_path,im_new);

def rescaleImAndSaveMeta(img_paths,meta_dir,power_scale_range=(-2,1),step_size=0.5):
	img_names=util.getFileNames(img_paths);
	power_range=np.arange(power_scale_range[0],power_scale_range[1]+1,step_size);
	scales=[2**val for val in power_range];

	scale_infos=[];
	for idx,scale in enumerate(scales):
		out_dir_curr=os.path.join(meta_dir,str(idx));
		util.mkdir(out_dir_curr);
		scale_infos.append((out_dir_curr,scale));

	args=[];
	idx=0;
	for idx_img,img_path in enumerate(img_paths):
		for out_dir_curr,scale in scale_infos:
			out_file=os.path.join(out_dir_curr,img_names[idx_img]);
			
			if os.path.exists(out_file):
				continue;

			args.append((img_path,out_file,scale,idx));
			idx=idx+1;

	p=multiprocessing.Pool(multiprocessing.cpu_count());
	p.map(rescaleImAndSave,args);

def script_writeValFile():
	dir_val='/disk2/ms_coco/val2014';
	out_dir='/disk2/mayExperiments/validation';
	util.mkdir(out_dir);

	imgs=util.getEndingFiles(dir_val,'.jpg');
	imgs=[os.path.join(dir_val,file_curr) for file_curr in imgs];
	imgs.sort();
	imgs=imgs[:5000];
	out_file=os.path.join(out_dir,'val.txt');
	util.writeFile(out_file,imgs)

def script_writeTrainFile():
	dir_val='/disk2/ms_coco/train2014';
	out_dir='/disk2/mayExperiments/train_data';
	util.mkdir(out_dir);

	imgs=util.getEndingFiles(dir_val,'.jpg');
	imgs=[os.path.join(dir_val,file_curr) for file_curr in imgs];
	imgs.sort();
	out_file=os.path.join(out_dir,'train.txt');
	util.writeFile(out_file,imgs)

def main():

	# out_dir_meta='/disk2/mayExperiments/validation';
	# val_file=os.path.join(out_dir_meta,'val.txt');
	out_dir_meta='/disk2/mayExperiments/train_data';
	val_file=os.path.join(out_dir_meta,'train.txt');
	out_dir=os.path.join(out_dir_meta,'rescaled_images');
	# util.mkdir(out_dir);

	img_paths=util.readLinesFromFile(val_file);
	# print len(img_paths);
	# img_paths=img_paths[:3];
	rescaleImAndSaveMeta(img_paths,out_dir);


if __name__=='__main__':
	main();

