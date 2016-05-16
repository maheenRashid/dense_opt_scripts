import os;
import util;

def main():
	file_list='/disk2/marchExperiments/dir_list.txt';
	out_file_commands='/disk2/marchExperiments/transfer_commands.txt';
	out_dir='/disk2/marchExperiments/hmdb/';
	pre_command='scp -r -i ~/.ssh/id_rsa_hpc1 maheenr@hpc1.engr.ucdavis.edu:'

	dirs_in=util.readLinesFromFile(file_list);
	commands=[];
	print len(dirs_in);
	for dir_in in dirs_in:
		dir_in=dir_in[:-1];
		dir_name=dir_in[dir_in.rindex('/')+1:];
		dir_out=os.path.join(out_dir,dir_name);
		if os.path.exists(dir_out):
			continue;
		else:
			# make command
			command=pre_command+dir_in+' '+out_dir;
			# append to commands;
			commands.append(command);
	for command in commands[:10]:
		print command;
	print len(commands);
	util.writeFile(out_file_commands,commands);


if __name__=='__main__':
	main();