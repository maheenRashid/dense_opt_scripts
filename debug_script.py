import scipy.misc;
import numpy as np;
import os;

def printImInfo(im):
	print im.shape;
	for i in range(im.shape[2]):
		print i,np.min(im[:,:,i]),np.max(im[:,:,i]);
	
def saveDummyLabelImage(path_to_tif,path_to_new_tif,dummy_val):

	im=scipy.misc.imread(path_to_tif);
	printImInfo(im);
	im[:,:,0]=dummy_val[0];
	im[:,:,1]=dummy_val[1];
	scipy.misc.imsave(path_to_new_tif,im);

	im_check=scipy.misc.imread(path_to_new_tif)
	printImInfo(im_check);

def main():
	train_dir='/home/maheenrashid/Downloads/debugging_jacob/opticalflow'
	data_dir=os.path.join(train_dir,'videos/v_BabyCrawling_g01_c01/images');
	path_to_tif=os.path.join(data_dir,'v_BabyCrawling_g01_c01.avi_000158.tif');
	path_to_new_tif=os.path.join(data_dir,'dummy_labels.tif');
	path_to_train_file=os.path.join(train_dir,'train_dummy.txt');

	dummy_val=[5,20];

	jpgs=[os.path.join(data_dir,file_curr) for file_curr in os.listdir(data_dir) if file_curr.endswith('.jpg')];
	with open(path_to_train_file,'wb') as f:
		for jpg_curr in jpgs:
			str_curr=jpg_curr+' '+path_to_new_tif+'\n';
			f.write(str_curr)
	


if __name__=='__main__':
	main();