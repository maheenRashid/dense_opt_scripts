import os;
import scipy.io;
import numpy as np;
import util;
import shutil;

def padZeros(num,num_zeros):
	in_file='0'*(num_zeros-len(str(num)));
	in_file=in_file+str(num);
	return in_file;
	
def moveFilesIntoFolders(in_dir,mat_file,out_dir,out_file_commands,pad_zeros_in=8,pad_zeros_out=4):
	arr=scipy.io.loadmat(mat_file)['ranges'];
	# videos=np.unique(arr);
	commands=[];
	for shot_no in range(arr.shape[1]):
		print shot_no,arr.shape[1];
		start_idx=arr[0,shot_no];
		end_idx=arr[1,shot_no];
		video_idx=arr[2,shot_no];
		out_dir_video=os.path.join(out_dir,str(video_idx));
		util.mkdir(out_dir_video);
		# print 
		# raw_input();
		shot_idx=np.where(shot_no==np.where(video_idx==arr[2,:])[0])[0][0]+1;
		out_dir_shot=os.path.join(out_dir_video,str(shot_idx));
		util.mkdir(out_dir_shot);

		# print start_idx,end_idx
		for idx,frame_no in enumerate(range(start_idx,end_idx+1)):
			in_file=os.path.join(in_dir,padZeros(frame_no,pad_zeros_in)+'.jpg');
			out_file=os.path.join(out_dir_shot,'frame'+padZeros(idx+1,pad_zeros_out)+'.jpg');
			command='mv '+in_file+' '+out_file;
			commands.append(command);
	print len(commands);
	util.writeFile(out_file_commands,commands);

def writeAviCommands(dirs,out_dir_videos,out_file_commands):
	command_pre='ffmpeg -f image2 -r 24 -i ';
	command_mid='/frame%04d.jpg -y ';
	commands=[];
	for dir_curr in dirs:
		for vid_dir in os.listdir(dir_curr):
			
			if not os.path.isdir(os.path.join(dir_curr,vid_dir)):
				continue;

			for shot_dir in os.listdir(os.path.join(dir_curr,vid_dir)):
				shot_dir_complete=os.path.join(dir_curr,vid_dir,shot_dir);
				if not os.path.isdir(shot_dir_complete):
					continue;
				out_file=dir_curr[dir_curr.rindex('/')+1:];
				out_file=out_file+'_'+str(vid_dir)+'_'+str(shot_dir)+'.avi';
				out_file=os.path.join(out_dir_videos,out_file);

				command=command_pre+shot_dir_complete+command_mid+out_file;
				commands.append(command);

	print len(commands)
	util.writeFile(out_file_commands,commands);




def main():
	dir_meta='/disk2/youtube_v2.2';
	dirs_class=['aeroplane','bird','boat','car','cat','cow','dog','horse','motorbike','train'];


	# dirs_class=['train']
	pre_range=os.path.join(dir_meta,'Ranges/ranges_');
	out_dir=os.path.join(dir_meta,'sorted_frames');

	dirs_combo=[os.path.join(out_dir,dir_class) for dir_class in dirs_class];
	
	out_dir_videos=os.path.join(dir_meta,'videos');
	out_file_commands=os.path.join(dir_meta,'make_avis.sh');

	util.mkdir(out_dir_videos);

	writeAviCommands(dirs_combo,out_dir_videos,out_file_commands)

	return
	util.mkdir(out_dir);

	for dir_class in dirs_class:
		out_dir_curr=	os.path.join(out_dir,dir_class);
		out_file_commands=os.path.join(out_dir,dir_class+'_mv_commands.txt');
		util.mkdir(out_dir_curr);
		moveFilesIntoFolders(os.path.join(dir_meta,dir_class),pre_range+dir_class+'.mat',out_dir_curr,out_file_commands);


if __name__=='__main__':
	main();