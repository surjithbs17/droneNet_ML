import os
import urllib3
import matplotlib.pyplot as plt
import numpy as np
import shutil
from scipy.misc import imresize

fig = plt.figure()

auth_string = 'arctic:domain'



def acquire_data(label,directory,start,end):
	if not os.path.exists(directory):
		os.makedirs(directory)


	for image in range(start,end):
		f = 'frame%08d.jpg' % image
		url = 'http://ecee.colorado.edu/~siewerts/softdefined_photom/media/DRONE-NET_TEST-10-29-16-FINAL/identified/'+ label + '/' + f
		print (url)
		http = urllib3.PoolManager()
		headers = urllib3.util.make_headers(basic_auth=auth_string)
		response = http.request('GET', url,headers = headers)
		path = directory + f
		with open(path, 'wb') as out:
			#print(response.data)
			out.write(response.data)
		response.release_conn()

		#img = plt.imread(os.path.join('database',f))

		#img = imresize(img, (100,100))
		#ax = fig.add_subplot(10,10,image)
		#img.astype(int)
		#plt.imshow(img)
		#plt.show()

def imcrop_tosquare(img):
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop


def get_img_array(start,end):
	np_file_p = 'numpy_files/%d_%d_p.npy'%(start,end)
	np_file_n = 'numpy_files/%d_%d_n.npy'%(start,end)
	if not os.path.exists(np_file_p):
		acquire_data('P','database/p/',start,end)
		p_filenames = [ 'database/p/frame%08d.jpg'%f for f in range(100)]	
		p_imgs = [plt.imread(fname)[..., :3] for fname in p_filenames]
		p_imgs = [imresize(img,(100,100)) for img in p_imgs ]
		p_imgs = np.array(p_imgs).astype(np.float32)
		shutil.rmtree('database/p/')
		if not os.path.exists('numpy_files/'):
			os.makedirs('numpy_files/')
		np.save(np_file_p,p_imgs)
	else:
		print('%d to %d P File already Exists'%(start,end))
		p_imgs = np.load(np_file_p)
	if not os.path.exists(np_file_n):
		acquire_data('N','database/n/',start,end)
		n_filenames = [ 'database/n/frame%08d.jpg'%f for f in range(100)]
		n_imgs = [plt.imread(fname)[..., :3] for fname in n_filenames]
		n_imgs = [imresize(img,(100,100)) for img in n_imgs ]
		n_imgs = np.array(n_imgs).astype(np.float32)
		shutil.rmtree('database/n/')
		if not os.path.exists('numpy_files/'):
			os.makedirs('numpy_files/')
		np.save(np_file_n,p_imgs)
	else:
		print('%d to %d N File already Exists'%(start,end))
		n_imgs = np.load(np_file_n)


	#print(p_imgs,n_imgs)
	print(p_imgs.shape,n_imgs.shape)
	

get_img_array(0,100)
#acquire_data('P','database',100)