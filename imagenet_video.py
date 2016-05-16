import numpy as np;
import util;
import os;
import script_debuggingFinetuning as sdf;
import multiprocessing;

def main():
	dir_meta='/disk3/maheen_data/imagenet_video/ILSVRC2015'
	mapping_file=os.path.join(dir_meta,'devkit/data/map_vid.txt');
	video_anno_meta=os.path.join(dir_meta,'ImageSet_initial','VID');
	path_to_vid=os.path.join(dir_meta,'Data/VID/snippets/train');
	out_dir_vids='/disk3/maheen_data/subset_for_training';
	util.mkdir(out_dir_vids);
	out_file_sh=os.path.join(out_dir_vids,'map_to_input.txt');
	num_to_pick=10;

	map_data=util.readLinesFromFile(mapping_file);
	map_data=[tuple(line.split(' ')) for line in map_data]
	[imagenet_id,class_id,string_id]=zip(*map_data);
	class_id=list(class_id);
	string_id=list(string_id);
	imagenet_id=list(imagenet_id);


	snippet_info_files=[os.path.join(video_anno_meta,file_curr) for file_curr in util.getStartingFiles(video_anno_meta,'train')];
	class_nums=[file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')] for file_curr in snippet_info_files];
	all_vids=[];
	vids={};
	for info_file_curr,class_num in zip(snippet_info_files,class_nums):
		lines=util.readLinesFromFile(info_file_curr);
		lines=[line[:line.index(' ')] for line in lines];
		all_vids=all_vids+lines;
		assert len(lines)==len(list(set(lines)));
		vids[class_num]=lines;

	black_list=[]
	for vid in all_vids:
		if all_vids.count(vid)>1:
			black_list.append(vid);

	# print len(black_list);
	for black_listed in black_list:
		for class_curr in vids.keys():
			if black_listed in vids[class_curr]:
				vids[class_curr].remove(black_listed);

	picked_vids={};
	for class_curr in vids.keys():
		vids_curr=[];
		for idx in range(num_to_pick):
			vid_curr=vids[class_curr][idx];
			vid_full_path=os.path.join(path_to_vid,vid_curr+'.mp4');
			assert os.path.exists(vid_full_path);
			vids_curr.append(vid_full_path);
		picked_vids[class_curr]=vids_curr;
		
	
	
	in_files=[];
	out_files=[];
	for class_curr in picked_vids.keys():
		# print class_curr,class_id.index(class_curr),string_id[class_id.index(class_curr)]
		file_pre=os.path.join(out_dir_vids,string_id[class_id.index(class_curr)]+'_')
		for vid_idx,vid_curr in enumerate(picked_vids[class_curr]):
			in_files.append(vid_curr);
			out_files.append(file_pre+str(vid_idx)+'.avi');

	# print len(in_files)
	# print len(out_files);
	# print in_files[0],out_files[0]
	args=zip(in_files,out_files);
	lines=[a+' '+b for a,b in args];
	util.writeFile(out_file_sh,lines);
	# print out_file_sh
	# sdf.shrinkVideos(args[0]);

	p=multiprocessing.Pool(multiprocessing.cpu_count());
	p.map(sdf.shrinkVideos,args);







	






if __name__=='__main__':
	main();