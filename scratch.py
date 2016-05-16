import numpy as np
import os;
import util;
import random;
import scipy.misc;
import cPickle as pickle;
import visualize;
import processOutput as po;
import scipy.io;

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


def main():
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

# def plotSimple(xAndYs,out_file,title='',xlabel='',ylabel='',legend_entries=None,loc=0,outside=False):






if __name__=='__main__':
	main();