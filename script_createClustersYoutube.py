import util;
import os;
import cPickle as pickle;
import copy;
import random;
import numpy as np;
import scipy.misc;

def pickFloImages(class_rec,flo_all,num_per_class,flo_jump=4):
	# sort them and make sure they're enough;
	flo_all_working=copy.deepcopy(flo_all);
	for idx,flo_curr in enumerate(flo_all_working):
		flo_curr.sort();
		flo_all_working[idx]=flo_curr[:-1*flo_jump];

	class_rec_dict=getClassFloDict(class_rec,flo_all_working);
	flos_picked=[];
	for class_curr in class_rec_dict.keys():
		flos_curr=class_rec_dict[class_curr]
		assert len(flos_curr)>=num_per_class;
		flos_picked_curr=random.sample(flos_curr, num_per_class);
		flos_picked.extend(flos_picked_curr);

	assert len(flos_picked)==num_per_class*len(class_rec_dict.keys())
	return flos_picked

def getClassFloDict(class_rec,flos_all):
	# put all of each category in a list.
	class_rec=np.array(class_rec);
	class_rec_uni=np.unique(class_rec);
	class_rec_dict={};
	for class_curr in class_rec_uni:
		idx_rel=np.where(class_rec==class_curr)[0];
		flos_curr=[flos_all[idx] for idx in idx_rel];
		flos_curr=[flo_curr for flo_list in flos_curr for flo_curr in flo_list];
		
		if class_curr not in class_rec_dict:
			class_rec_dict[class_curr]=[];	
		class_rec_dict[class_curr].extend(flos_curr);
	return class_rec_dict;

def getFloInfo(dir_meta,out_file_list,video_names=None):
	if video_names is None:
		video_list=util.getFilesInFolder(dir_meta,'.avi');
		video_names=util.getFileNames(video_list,ext=False);
	flo_all=[];
	class_rec=[];
	flo_count_rec=[];
	dir_rec=[];
	for idx_video_name,video_name in enumerate(video_names):
		print idx_video_name,len(video_names)
		dir_curr=os.path.join(dir_meta,video_name);
		if not os.path.exists(dir_curr):
			continue;
		flo_list=util.getFilesInFolder(dir_curr,'.flo');
		flo_all.append(flo_list);
		dir_rec.append(dir_curr);
		class_rec.append(video_name[:video_name.index('_')]);
		flo_count_rec.append(len(flo_list));
	print len(dir_rec),len(class_rec),len(flo_count_rec),len(flo_all)

	pickle.dump([dir_rec,class_rec,flo_count_rec,flo_all],open(out_file_list,'wb'));


def getFloPaths(flo_path,flo_jump=4):
	flos=[flo_path];

	flo_bef=flo_path[:flo_path.rindex('_')+1];
	flo_num=flo_path[flo_path.rindex('_')+1:flo_path.rindex('.')];
	flo_aft=flo_path[flo_path.rindex('.'):];

	str_len=len(flo_num);
	num=int(flo_num);
	for add_val in range(1,flo_jump+1):
		str_new=str(num+add_val);
		zero_pad='0'*(str_len-len(str_new));
		str_new=zero_pad+str_new;
		str_new=flo_bef+str_new+flo_aft;
		flos.append(str_new);

	return flos;

def resize(flo,im_shape):
    gt_flo_sp=np.zeros((im_shape[0],im_shape[1],2));

    for layer_idx in range(flo.shape[2]):
        min_layer=np.min(flo[:,:,layer_idx]);
        max_layer=np.max(flo[:,:,layer_idx]);
        gt_flo_sp_curr=scipy.misc.imresize(flo[:,:,layer_idx],im_shape);
        gt_flo_sp_curr=gt_flo_sp_curr/float(max(np.max(gt_flo_sp_curr),np.finfo(float).eps));
        gt_flo_sp_curr=gt_flo_sp_curr*(max_layer-min_layer);
        gt_flo_sp_curr=gt_flo_sp_curr+min_layer;
        gt_flo_sp[:,:,layer_idx]=gt_flo_sp_curr;

    return gt_flo_sp;	

def addFlowsAndDownScale(flo_paths,downscale=(20,20)):
	flos=[];
	for flo_path in flo_paths:
		flo=util.readFlowFile(flo_path);
		flos.append(flo);
	flos=np.array(flos);
	flos=np.sum(flos,axis=0);
	flos=resize(flos,tuple(downscale));
	return flos;

def makeClusters():
	pass;

def main():

	dir_meta='/group/leegrp/maheen_data/youtube_train_rs';
	out_file_list=os.path.join(dir_meta,'flo_info.p');
	
	video_list=util.getFilesInFolder(dir_meta,'.avi');
	video_names=util.getFileNames(video_list,ext=False);	
	getFloInfo(dir_meta,out_file_list,video_names)
	# [dir_rec,class_rec,flo_count_rec,flo_all]=pickle.load(open(out_file_list,'rb'));

	# flos_picked=pickFloImages(class_rec,flo_all,10);
	# print len(flos_picked);
	# print flos_picked[:10];		



if __name__=='__main__':
	main();