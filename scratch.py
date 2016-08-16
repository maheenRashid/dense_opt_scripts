import numpy as np
import os;
import util;
import random;
import scipy.misc;
import cPickle as pickle;
import visualize;
import processOutput as po;
import scipy.io;
import scipy.stats;
import cv2
def getTifsFromTrain(old_train_txt):
	tif_list_old=util.readLinesFromFile(old_train_txt);
	tif_list_old=[line[line.index(' ')+1:] for line in tif_list_old];
	return tif_list_old;

def getDirMetaFromTifPath(tif_path):
	dir_meta_old=tif_path.split('/');
	dir_meta_old=dir_meta_old[:-3];
	dir_meta_old='/'.join(dir_meta_old);
	return dir_meta_old;

def getImageDirFromTifPath(tif_path):
	image_dir=tif_path.split('/');
	image_dir=image_dir[-2];
	return image_dir;

def script_sanityCheckEquivalenceClusters():
	old_train_txt='/disk2/mayExperiments/ft_youtube_hmdb_newClusters/train.txt';
	new_train_txt='/disk3/maheen_data/ft_youtube_40/train.txt';


	tif_list_old=getTifsFromTrain(old_train_txt);
	tif_list_new=getTifsFromTrain(new_train_txt);
	
	print len(tif_list_old),len(tif_list_new);

	dir_meta_old=getDirMetaFromTifPath(tif_list_old[0]);
	dir_meta_new=getDirMetaFromTifPath(tif_list_new[0]);
	image_dir_old=getImageDirFromTifPath(tif_list_old[0]);
	image_dir_new=getImageDirFromTifPath(tif_list_new[0]);


	print dir_meta_old,dir_meta_new,image_dir_old,image_dir_new

	# return 
	im_names_old=util.getFileNames(tif_list_old);
	im_names_new=util.getFileNames(tif_list_new);

	tif_list_both=list(set(im_names_old).intersection(set(im_names_new)));
	
	print len(tif_list_both)

	num_to_pick=100;
	random.shuffle(tif_list_both);
	tif_list_both=tif_list_both[:num_to_pick];
	for tif_name in tif_list_both:
		video_name=tif_name[:tif_name.index('.')];
		old_tif_path=os.path.join(dir_meta_old,video_name,image_dir_old,tif_name);
		new_tif_path=os.path.join(dir_meta_new,video_name,image_dir_new,tif_name);
		
		tif_new=scipy.misc.imread(new_tif_path);
		tif_old=scipy.misc.imread(old_tif_path);

		assert np.array_equal(tif_new,tif_old);

def script_visualizeRatios():
	ratio_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/ratios.p';
	out_file_plot='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/ratios_plot.png';

	ratio=pickle.load(open(ratio_file,'rb'));
	print ratio.keys();
	xAndYs=[];
	legend_entries=[];
	for key_curr in ratio.keys():
		print key_curr,np.array(ratio[key_curr]).shape;
		list_curr=np.array(ratio[key_curr]);
		index_nan=np.min(np.where(np.isnan(list_curr)));
		assert np.sum(np.isnan(list_curr[index_nan:]))==list_curr[index_nan:].size
		list_curr=list_curr[:index_nan];
		list_curr=list_curr[:100]
		xAndYs.append((range(len(list_curr)),list_curr));
		legend_entries.append(key_curr);

	visualize.plotSimple(xAndYs,out_file_plot,'update/weight ratio','iterations','ratio',legend_entries,0,True);

def script_compareClusters():
	clusters_me='/disk2/mayExperiments/youtube_subset_new_cluster/clusters.mat'
	clusters_j='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
	out_file='/disk2/temp/clusters_comp.png';
	# clusters_me=scipy.io.loadMat(cluters_me);
	
	clusters_me=scipy.io.loadmat(clusters_me)['C'];
	mags_me=np.linalg.norm(clusters_me,axis=1);
	print mags_me.shape;
	print np.mean(mags_me);

	# clusters_me=clusters_me*4;
	clusters_j=po.readClustersFile(clusters_j);
	mags_j=np.linalg.norm(clusters_j,axis=1);
	print mags_j.shape;
	print np.mean(mags_j);
	return
	print clusters_me
	print clusters_j
	xAndYs=[(clusters_me[:,0],clusters_me[:,1]),(clusters_j[:,0],clusters_j[:,1])]
	visualize.plotScatter(xAndYs,out_file,color=['r','b']);

def script_seeMultipleClusters(dir_clusters=None,out_dir_plot=None):
	if dir_clusters is None:
		dir_clusters='/disk3/maheen_data/debug_networks/clusters_youtube_multiple';

	if out_dir_plot is None:
		out_dir_plot='/disk2/temp/cluster_plots';

	util.mkdir(out_dir_plot);	
	clusters_all=util.getFilesInFolder(dir_clusters,'.npy');
	print len(clusters_all);
	for idx_cluster_file,cluster_file in enumerate(clusters_all):
		print idx_cluster_file;
		cluster_name=util.getFileNames([cluster_file],ext=False)[0];
		out_file=os.path.join(out_dir_plot,cluster_name+'.png');
		cluster_curr=np.load(cluster_file);
		visualize.plotScatter([(cluster_curr[:,0],cluster_curr[:,1])],out_file,color='r');		
		# files_all.append(out_file);

	visualize.writeHTMLForFolder(out_dir_plot,ext='.png',height=300,width=300);

def scaleAndSingleTif(tif):
	tif=tif[:,:,0];
	tif=tif*255;
	tif=np.dstack((tif,tif,tif));
	return tif;


def script_checkSuppressFlowMatlabCode():
	dir_meta='/disk2/temp/aeroplane_10_3';
	dir_bef=os.path.join(dir_meta,'noThresh');
	dir_aft=os.path.join(dir_meta,'withThresh');
	out_dir=os.path.join(dir_meta,'viz');
	util.mkdir(out_dir)

	tif_files=util.getFilesInFolder(dir_bef,'.tif');
	tif_files=util.getFileNames(tif_files);
	for file_curr in tif_files:
		file_bef=os.path.join(dir_bef,file_curr);
		file_aft=os.path.join(dir_aft,file_curr);
		
		tif_bef_one=scipy.misc.imread(file_bef);
		tif_bef_one=tif_bef_one[:,:,0];

		tif_aft_one=scipy.misc.imread(file_aft);
		tif_aft_one=tif_aft_one[:,:,0];
		
		mat_info_bef=scipy.io.loadmat(os.path.join(dir_bef,file_curr[:file_curr.rindex('.')]+'.mat'));
		R=mat_info_bef['R'];
		L=mat_info_bef['L'];
		# print optFlow.shape,R.shape,L.shape
		# optFlow=np.dstack((optFlow,np.zeros((optFlow.shape[0],optFlow.shape[1],1))));
		# optFlow=cv2.resize(optFlow,(20,20));
		# optFlow=cv2.resize(optFlow,(R.shape[1],R.shape[0]));

		# print optFlow.shape
		# mag_bef_o=np.power(np.power(optFlow[:,:,0],2)+np.power(optFlow[:,:,1],2),0.5);
		mag_bef=np.power(np.power(R,2)+np.power(L,2),0.5);
		
		idx=np.where(mag_bef<1.0);
		# print idx[0].shape;
		# idx_o=np.where(mag_bef_o<1.0);
		# print idx_o[0].shape
		# print np.setdiff1d(idx[0],idx_o[0])
		# print np.setdiff1d(idx[1],idx_o[1])

		# break;


		print 'BEFORE'
		print np.unique(R[idx]);
		print np.unique(tif_bef_one[idx]);

		print 'AFTER'
		mat_info_aft=scipy.io.loadmat(os.path.join(dir_aft,file_curr[:file_curr.rindex('.')]+'.mat'));
		print np.unique(mat_info_aft['R'][idx]);
		print np.unique(tif_aft_one[idx]);
		assert np.unique(tif_aft_one[idx])[0]==40;

		
def script_makeUCFTestTrainTxt():
	dir_meta='/home/maheenrashid/Downloads/opticalflow/videos/v_BabyCrawling_g01_c01/images';
	out_dir='/disk3/maheen_data/debug_networks/sanityCheckDebug';
	util.mkdir(out_dir);

	train_file=os.path.join(out_dir,'train.txt');

	tifs=util.getFilesInFolder(dir_meta,'.tif');
	imgs=[file_curr.replace('.tif','.jpg') for file_curr in tifs];
	for file_curr in imgs:
		assert os.path.exists(file_curr)
	lines=[img+' '+tif for img,tif in zip(imgs,tifs)];

	util.writeFile(train_file,lines);


def main():
	
	dir_clusters='/disk2/temp/youtube_clusters_check_nothresh';
	out_dir=os.path.join(dir_clusters,'viz');
	util.mkdir(out_dir);
	script_seeMultipleClusters(dir_clusters,out_dir)

	





	return
	
	script_seeMultipleClusters();
	return
	dir_clusters='/disk3/maheen_data/debug_networks/clusters_youtube_multiple';
	clusters_all=util.getFilesInFolder(dir_clusters,'.npy');
	clusters_all=[file_curr for file_curr in clusters_all if 'harder' in file_curr];
	clusters_all.append(os.path.join(dir_clusters,'clusters_original.npy'));
	min_mags=[];
	for file_curr in clusters_all:
		clusters=np.load(file_curr);
		mags=np.power(np.sum(np.power(clusters,2),axis=1),0.5);
		min_mag=np.min(mags);
		min_mags.append(min_mag);

	print min_mags,np.max(min_mags);

	thresh=1;
	counts=[];
	for file_curr in clusters_all:
		clusters=np.load(file_curr);
		print file_curr
		mags=np.power(np.sum(np.power(clusters,2),axis=1),0.5);
		count=np.sum(mags<=thresh);
		print count
		counts.append(count);

	print np.mean(counts);


	# return
	dir_curr='/disk3/maheen_data/debug_networks/figuringClustering';
	mag_file=os.path.join(dir_curr,'mags_all.npy');
	mags=np.load(mag_file);
	
	print len(mags),np.sum(mags<=thresh);
	mags=mags[mags>thresh];
	print len(mags);

	out_file=os.path.join(dir_curr,'mag_hist_noZero.png');
	visualize.hist(mags,out_file,bins=40,normed=True,xlabel='Value',ylabel='Frequency',title='',cumulative=False);
	print out_file.replace('/disk3','vision3.cs.ucdavis.edu:1001');

	print np.min(mags),np.max(mags),np.mean(mags),np.std(mags);
	

	
# def plotSimple(xAndYs,out_file,title='',xlabel='',ylabel='',legend_entries=None,loc=0,outside=False):






if __name__=='__main__':
	main();