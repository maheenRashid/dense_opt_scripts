import numpy as np;
import util
import os;
import scipy.misc;
import scipy.stats;
import processOutput as po;
import visualize;
import random;
import multiprocessing;
import cv2;
import scipy.spatial.distance;
import subprocess;

def getCorrespondingIm(flo_path,pre_path,ext='.jpg',post_path='images_transfer'):
	file_name=util.getFileNames([flo_path],ext=False)[0];
	video_name=file_name[:file_name.index('.')];
	dir_curr=os.path.join(pre_path,video_name,post_path);
	im=os.path.join(dir_curr,file_name+ext);
	return im;

def replaceClusterIdWithFlow(tif,clusters):
	tif_dim=np.zeros(tif.shape);
	for dim in range(2):
		val_tif=np.unique(tif[:,:,dim].ravel());
		for val_curr in val_tif:
			tif_dim[tif[:,:,dim]==val_curr,dim]=clusters[val_curr-1,dim]
		# assert len(np.unique(tif_dim[:,:,dim]))==len(np.unique(tif[:,:,dim]));
	return tif_dim;

def script_writeFloVizHTML(out_file_html,out_dir_viz,flo_files,im_files,tif_files,clusters,tifAsPng=False):

	img_paths=[];
	captions=[];
	# idx=0;
	for flo_file,im_file,tif_file in zip(flo_files,im_files,tif_files):
		# print idx;
		# print tif_file
		assert os.path.exists(tif_file);
		assert os.path.exists(im_file);
		# print tif_file
		# if not os.path.exists(tif_file) or not os.path.exists(im_file) :
		# 	continue;
		file_name=util.getFileNames([flo_file],ext=False)[0];
		out_file_pre=os.path.join(out_dir_viz,file_name);
		out_file_flo_viz=out_file_pre+'_flo.png';
		out_files_tif=[out_file_pre+'_tifim_x.png',out_file_pre+'_tifim_y.png',out_file_pre+'_tifflo.png'];
		if not os.path.exists(out_file_flo_viz):
			flo=util.readFlowFile(flo_file);
			po.saveFloFileViz(flo_file,out_file_flo_viz);
		for idx,out_file_tif_viz in enumerate(out_files_tif):
			tif=scipy.misc.imread(tif_file)[:,:,:2];
			if idx==0 and not os.path.exists(out_file_tif_viz):
				tif_flo=replaceClusterIdWithFlow(tif,clusters);
				po.saveMatFloViz(tif_flo,out_file_tif_viz);

			if not os.path.exists(out_file_tif_viz) and idx==1:
				tif_x=np.array(tif[:,:,0]*(255.0/clusters.shape[0]),dtype=int);
				tif_x=np.dstack((tif_x,tif_x,tif_x));
				scipy.misc.imsave(out_file_tif_viz,tif_x);

			if not os.path.exists(out_file_tif_viz) and idx==2:
				tif_x=np.array(tif[:,:,1]*(255.0/clusters.shape[0]),dtype=int);
				tif_x=np.dstack((tif_x,tif_x,tif_x));
				scipy.misc.imsave(out_file_tif_viz,tif_x);

			
		img_paths_curr=[im_file,out_file_flo_viz]+out_files_tif;
		im_name=util.getFileNames([im_file],ext=False)[0];
		captions_curr=[im_name,'flo_viz']+['tif_flo_viz']*len(out_files_tif)

		# if tifAsPng:
		# 	img_paths_curr.append(out_file_tif_viz.replace('_x.png','_y.png'));
		# 	captions_curr.append('tif_flo_viz');

		img_paths_curr=[util.getRelPath(file_curr) for file_curr in img_paths_curr];
		img_paths.append(img_paths_curr);
		captions.append(captions_curr);
		# idx=idx+1;

	visualize.writeHTML(out_file_html,img_paths,captions)

def saveTifGray(tif,out_file_x,out_file_y,num_clusters):
	tif_x=np.array(tif[:,:,0]*(255.0/num_clusters),dtype=int);
	tif_x=np.dstack((tif_x,tif_x,tif_x));
	scipy.misc.imsave(out_file_x,tif_x);

	tif_y=np.array(tif[:,:,1]*(255.0/num_clusters),dtype=int);
	tif_y=np.dstack((tif_y,tif_y,tif_y));
	scipy.misc.imsave(out_file_y,tif_y);

def recordContainingFiles(dirs,num_to_evaluate,out_file_hmdb,post_dir='images',ext='.flo'):
	random.shuffle(dirs);
	print len(dirs);
	dirs=dirs[:num_to_evaluate];
	print dirs[0]
	tifs=[];
	for idx_dir_curr,dir_curr in enumerate(dirs):
		print idx_dir_curr
		tif_files=[os.path.join(dir_curr,file_curr) for file_curr in util.getFilesInFolder(os.path.join(dir_curr,post_dir),ext)];
		tifs.extend(tif_files);
	print len(tifs)
	util.writeFile(out_file_hmdb,tifs);


def makeTifHists(tif_files,out_file_x,out_file_y,bins=range(1,41)):
	tif_x=[];
	tif_y=[];

	for idx_file_curr,file_curr in enumerate(tif_files):
		print idx_file_curr
		tif_curr=scipy.misc.imread(file_curr);
		tif_curr_x=list(np.ravel(tif_curr[:,:,0]));
		tif_curr_y=list(np.ravel(tif_curr[:,:,1]));
		tif_x=tif_x+tif_curr_x;
		tif_y=tif_y+tif_curr_y;
	len(tif_x);
	len(tif_y);
	print scipy.stats.mode(tif_x);
	print scipy.stats.mode(tif_y);
	visualize.hist(tif_x,out_file_x,bins=bins)
	visualize.hist(tif_y,out_file_y,bins=bins)


def script_vizForHMDB():

	out_dir='/disk2/mayExperiments/debug_finetuning/hmdb';
	clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
	vid_list=os.path.join(out_dir,'video_list.txt');
	out_dir_viz=os.path.join(out_dir,'im');
	util.mkdir(out_dir_viz);
	out_file_html=out_dir_viz+'.html';
	
	path_to_hmdb='/disk2/marchExperiments/hmdb_try_2/hmdb'

	dirs=util.readLinesFromFile(vid_list);
	dirs=[os.path.join(path_to_hmdb,dir_curr) for dir_curr in dirs[2:]];
	random.shuffle(dirs);
	num_to_evaluate=100;
	out_file_tif=os.path.join(out_dir,'tif_list.txt');

	# recordContainingFiles(dirs,num_to_evaluate,out_file_flo,post_dir='images',ext='.flo');
	tif_files=util.readLinesFromFile(out_file_tif);
	tif_files=tif_files[:100];
	img_files=[file_curr.replace('.tif','.jpg') for file_curr in tif_files];
	flo_files=[file_curr.replace('.tif','.flo') for file_curr in tif_files];
	clusters=po.readClustersFile(clusters_file);

	script_writeFloVizHTML(out_file_html,out_dir_viz,flo_files,img_files,tif_files,clusters,True)

	# out_file_x=os.path.join(out_dir_viz,'x_hist.png');
	# out_file_y=os.path.join(out_dir_viz,'y_hist.png');

	# makeTifHists(tif_files,out_file_x,out_file_y);

def reshapeFloFiles(flo_files,tif_files,out_dir_new_flos):
	for flo_file,tif_file in zip(flo_files,tif_files):
		# print flo_file
		flo=util.readFlowFile(flo_file);
		# print flo.shape
		tif=scipy.misc.imread(tif_file);
		# print tif.shape;

		flo_rs=cv2.resize(flo,(tif.shape[1],tif.shape[0]));
		# print flo_rs.shape

		flo_rs[:,:,0]=flo_rs[:,:,0]*(tif.shape[0]/float(flo.shape[0]));
		flo_rs[:,:,1]=flo_rs[:,:,1]*(tif.shape[1]/float(flo.shape[1]));
 		flo_rs=flo_rs*5;

 	# 	print np.min(flo_rs[:,:,0]),np.max(flo_rs[:,:,1])
		# print np.min(flo[:,:,0]),np.max(flo[:,:,1])

 		flo_name=util.getFileNames([flo_file],ext=True)[0];
 		# print flo_name
 		out_file_curr=os.path.join(out_dir_new_flos,flo_name);
 		util.writeFlowFile(flo_rs,out_file_curr);


def shrinkVideos((inpath,out_path)):
	if os.path.exists(out_path):
		return;

	command='ffmpeg -i ';
	command=command+inpath;
	command=command+' -vf scale=320:240 ';
	command=command+out_path;
	print command
	subprocess.call(command,shell=True);

	# /disk2/youtube_v2.2/videos/dog_13_5.avi -vf scale=320:240 /disk2/mayExperiments/youtube_subset/dog_13_5.avi


def makeImTifViz(img_paths_all,tif_paths_all,out_file_html,out_dir_tif,num_clusters=40,disk_path='/disk2'):
	out_files_tif_x=[os.path.join(out_dir_tif,img_name+'_x.png') for img_name in util.getFileNames(tif_paths_all,ext='False')];
	out_files_tif_y=[os.path.join(out_dir_tif,img_name+'_y.png') for img_name in util.getFileNames(tif_paths_all,ext='False')];
	

	for tif_path,out_file_x,out_file_y in zip(tif_paths_all,out_files_tif_x,out_files_tif_y):
		tif=scipy.misc.imread(tif_path);
		# print np.min(tif[:,:,:2]),np.max(tif[:,:,:2])
		assert np.min(tif[:,:,:2])>0 and np.max(tif[:,:,:2])<num_clusters+1;
		saveTifGray(tif,out_file_x,out_file_y,num_clusters)
	
	# out_file_html=out_dir_tif+'.html';
	img_paths_html=[[util.getRelPath(img_curr,disk_path) for img_curr in img_list] for img_list in zip(img_paths_all,out_files_tif_x,out_files_tif_y)];
	# captions_html=[[util.getFileNames([img_curr],ext=False)[0] for img_curr in img_list] for img_list in zip(img_paths_all,out_files_tif_x,out_files_tif_y)];
	captions_html=[['Image','Tif_x','Tif_y']]*len(img_paths_html);
	visualize.writeHTML(out_file_html,img_paths_html,captions_html);


def script_findMinCluster(clusters_file,new_flag=False):
	if new_flag:
		clusters=scipy.io.loadmat(clusters_file);
		clusters=clusters['C']
	else:
		clusters=po.readClustersFile(clusters_file);

	print clusters.shape;
	norms=np.linalg.norm(clusters,axis=1);
	min_idx=np.argmin(norms);

	print 'MIN INFO', min_idx,norms[min_idx],clusters[min_idx,:]
	return min_idx,clusters;

def main():
	train_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/train.txt'
	files=util.readLinesFromFile(train_file);
	random.shuffle(files);
	files=files[:100];
	img_paths_all=[line[:line.index(' ')] for line in files];
	tif_paths_all=[line[line.index(' ')+1:] for line in files];
	num_clusters=40;

	out_dir='/disk3/maheen_data/debug_networks';
	util.mkdir(out_dir);

	out_dir_tif=os.path.join(out_dir,'tif');
	util.mkdir(out_dir_tif);
	out_file_html=os.path.join(out_dir,'tif_suppressCluster.html');

	out_files_tif_x=[os.path.join(out_dir_tif,img_name+'_x.png') for img_name in util.getFileNames(tif_paths_all,ext='False')];
	out_files_tif_y=[os.path.join(out_dir_tif,img_name+'_y.png') for img_name in util.getFileNames(tif_paths_all,ext='False')];
	

	for tif_path,out_file_x,out_file_y in zip(tif_paths_all,out_files_tif_x,out_files_tif_y):
		# print tif_path
		tif=scipy.misc.imread(tif_path);
		# print np.min(tif[:,:,:2]),np.max(tif[:,:,:2])
		assert np.min(tif[:,:,:2])>0 and np.max(tif[:,:,:2])<num_clusters+1;
		saveTifGray(tif,out_file_x,out_file_y,num_clusters)
	

	makeImTifViz(img_paths_all,tif_paths_all,out_file_html,out_dir_tif,num_clusters=40,disk_path='disk3')


	return
	clusters_file='/disk2/mayExperiments/youtube_subset_new_cluster/clusters.mat';
	clusters_ucf='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
	min_idx_new,C_new=script_findMinCluster(clusters_file,new_flag=True);

	min_idx_ucf,C_ucf=script_findMinCluster(clusters_ucf,new_flag=False);
	print min_idx_new,min_idx_ucf
	




	return
	out_dir='/disk2/mayExperiments/imagenet_subset'
	out_file_html=out_dir+'.html';
	img_paths=util.getFilesInFolder(out_dir,'.jpg');
	tif_paths=[img_path.replace('.jpg','.tif') for img_path in img_paths]
	out_dir_tif=os.path.join(out_dir,'tif_viz');
	util.mkdir(out_dir_tif);
	makeImTifViz(img_paths,tif_paths,out_file_html,out_dir_tif);

	return
	train_txt='/disk2/mayExperiments/ft_youtube_hmdb_ucfClusters/train.txt';
	out_dir='/disk2/mayExperiments/eval_ucf_finetune';
	out_dir_tif=os.path.join(out_dir,'train_tif_select');

	train_txt='/disk2/mayExperiments/ft_youtube_hmdb_newClusters/train.txt';
	out_dir='/disk2/mayExperiments/eval_newClusters_finetune';
	util.mkdir(out_dir);
	out_dir_tif=os.path.join(out_dir,'train_tif_select');
	util.mkdir(out_dir_tif);

	num_to_pick=20;
	num_clusters=40;
	train_data=util.readLinesFromFile(train_txt);
	img_paths=[line_curr[:line_curr.index(' ')] for line_curr in train_data];
	tif_paths=[line_curr[line_curr.index(' ')+1:] for line_curr in train_data];
	print img_paths[0].split('/');
	# return
	dataset=np.array([img_path.split('/')[4] for img_path in img_paths]);
	print np.unique(dataset);

	idx_youtube=np.where(dataset=='youtube')[0];
	classes_idx=[];
	classes_idx.append(np.where(dataset!='youtube')[0]);
	img_paths_youtube=list(np.array(img_paths)[idx_youtube]);
	img_paths_youtube_classes=np.array([img_path[:img_path.index('_')] for img_path in util.getFileNames(img_paths_youtube)])
	for class_curr in np.unique(img_paths_youtube_classes):
		idx_rel=np.where(img_paths_youtube_classes==class_curr)[0];
		class_idx_org=idx_youtube[idx_rel];
		classes_idx.append(class_idx_org);

	# print len(idx_youtube);
	for idx,class_idx in enumerate(classes_idx):
		# print len(class_idx);
		if idx>0:
			paths=np.array(img_paths)[class_idx];
			dataset=[img_name[:img_name.index('_')] for img_name in util.getFileNames(paths)];
			# print set(dataset);
			assert len(set(dataset))==1

	img_paths_all=[];
	tif_paths_all=[];
	for class_idx in classes_idx:
		img_paths_rel=np.array(img_paths)[class_idx[:num_to_pick]];
		tif_paths_rel=np.array(tif_paths)[class_idx[:num_to_pick]];
		img_paths_all=img_paths_all+list(img_paths_rel);
		tif_paths_all=tif_paths_all+list(tif_paths_rel);


	out_files_tif_x=[os.path.join(out_dir_tif,img_name+'_x.png') for img_name in util.getFileNames(tif_paths_all,ext='False')];
	out_files_tif_y=[os.path.join(out_dir_tif,img_name+'_y.png') for img_name in util.getFileNames(tif_paths_all,ext='False')];
	

	for tif_path,out_file_x,out_file_y in zip(tif_paths_all,out_files_tif_x,out_files_tif_y):
		# print tif_path
		tif=scipy.misc.imread(tif_path);
		# print np.min(tif[:,:,:2]),np.max(tif[:,:,:2])
		assert np.min(tif[:,:,:2])>0 and np.max(tif[:,:,:2])<num_clusters+1;
		saveTifGray(tif,out_file_x,out_file_y,num_clusters)
	out_file_html=out_dir_tif+'.html';
	img_paths_html=[[util.getRelPath(img_curr) for img_curr in img_list] for img_list in zip(img_paths_all,out_files_tif_x,out_files_tif_y)];
	# captions_html=[[util.getFileNames([img_curr],ext=False)[0] for img_curr in img_list] for img_list in zip(img_paths_all,out_files_tif_x,out_files_tif_y)];
	captions_html=[['Image','Tif_x','Tif_y']]*len(img_paths_html);
	visualize.writeHTML(out_file_html,img_paths_html,captions_html);
		




if __name__=='__main__':
	main();
