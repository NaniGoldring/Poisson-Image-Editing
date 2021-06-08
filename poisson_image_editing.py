import numpy as np
import cv2
import scipy 
import scipy.sparse as sps
import scipy.sparse.linalg as linalg

# linear least squares solver
def linlsq_solver(A, b, dims):
	x = linalg.spsolve(A.tocsc(),b)
	return np.reshape(x,(dims[0],dims[1]))

# stitches poisson equation solution with target
def stitch_images(source, target, dims):
	target[dims[0]:dims[1], dims[2]:dims[3],:] = source
	return target

# performs poisson blending
def blend_image(source, target, mask, offset, GRAD_MIX):
	equation_param = []
	ch_data = {}
	
	M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
	
	# construct poisson equation 
	for ch in range(3):
		ch_source = cv2.warpAffine(source[:,:,ch],M,(target.shape[1],target.shape[0]))
		ch_mask = mask 
		ch_target = target[:,:,ch]

		equation_param.append(poisson_blending(ch_source, ch_mask, ch_target, GRAD_MIX))

	# solve poisson equation
	image_solution = np.empty_like(target)
	for i in range(3):
		image_solution[:,:,i] = linlsq_solver(equation_param[i][0],equation_param[i][1],target.shape)
 
 
	return image_solution


def poisson_blending(source, mask, target, GRAD_MIX):
	# comparison function
	def _compare(val1, val2):
		if(abs(val1) > abs(val2)):
			return val1
		else:
			return val2

	# membrane (region where Poisson blending is performed)
	#mask = image['mask']
	Hs,Ws = mask.shape
	
	num_pxls = Hs * Ws
	

	# source and target image
	source = source.flatten(order='C')
	target = target.flatten(order='C')

	# initialise the mask, guidance vector field and laplacian
	mask = mask.flatten(order='C')
	guidance_field = np.zeros(mask.shape,dtype='float64')
	laplacian = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')
	Np_up_s = 0
	Np_left_s = 0
	Np_down_s = 0
	Np_right_s = 0
	for i in range(num_pxls):
	# construct the sparse laplacian block matrix
	# and guidance field for the membrane
		if(mask[i] != 0):
	
			laplacian[i, i] = 4
		
			# construct laplacian, and compute source and target gradient in mask
			if(i - Ws > 0):
				laplacian[i, i-Ws] = -1
				Np_up_s = source[i] - source[i-Ws]
				Np_up_t = target[i] - target[i-Ws]
			else:
				Np_up_s = source[i]
				Np_up_t = target[i]
		
			if(i % Ws != 0):
				laplacian[i, i-1] = -1
				Np_left_s = source[i] - source[i-1]
				Np_left_t = target[i] - target[i-1]
			else:
				Np_left_s = source[i]
				Np_left_t = target[i]
		
			if(i + Ws < num_pxls):
				laplacian[i, i+Ws] = -1
				Np_down_s = source[i] - source[i+Ws]
				Np_down_t = target[i] - target[i+Ws]
			else:
				Np_down_s = source[i]
				Np_down_t = target[i]
		
			if(i % Ws != Ws-1):
				laplacian[i, i+1] = -1
				Np_right_s = source[i] - source[i+1]
				Np_right_t = target[i] - target[i+1]
			else:
				Np_right_s = source[i]
				Np_right_t = target[i]
			
			# choose stronger gradient
			if(GRAD_MIX is False):
				Np_up_t = 0
				Np_left_t = 0
				Np_down_t = 0
				Np_right_t = 0
			
			guidance_field[i] = (_compare(Np_up_s, Np_up_t) + _compare(Np_left_s, Np_left_t) + 
					_compare(Np_down_s, Np_down_t) + _compare(Np_right_s, Np_right_t))

		else:
			# if point lies outside membrane, copy target function
			laplacian[i, i] = 1
			guidance_field[i] = target[i]

	return [laplacian, guidance_field]


