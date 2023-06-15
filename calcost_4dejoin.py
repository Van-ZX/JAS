from ast import Lambda
from platform import python_branch
from tkinter.tix import IMAGE
import numpy as np
from scipy import ndimage, signal
# import matlab
# import matlab.engine
# eng = matlab.engine.start_matlab()

IMAGE_SIZE = 512

def getHt(p):
	p[p == 0] = 1e-10

	H = - (p * np.log2(p))
	# H((P < np.spacing(1)) | (P > 1 - np.spacing(1))) = 0
	H[(p < 2.2204e-16) | (p > 1 - 2.2204e-16)] = 0

	Ht = H.sum()

	return Ht

def getP(Lambda, cost):
	# import ipdb
	# ipdb.set_trace()
	
	JointProb = np.exp(-Lambda * cost)
	A = np.ones([cost.shape[0], 1], dtype=np.float)
	sum = np.sum(JointProb, axis=0).reshape(1, -1)
	A = np.dot(A, sum)
	p = np.zeros(cost.shape, dtype=np.float)
	p = JointProb / A

	return p

def calc_lambda(cost, message_length, n):
	l3 = 1e+3
	m3 = message_length + 1
	iterations = 0
	while m3 > message_length:
		l3 = l3 * 2
		p_dejoin = getP(l3, cost)
		m3 = getHt(p_dejoin)
		iterations = iterations + 1
		if iterations > 30:
			Lambda = l3
			return Lambda

	l1 = 0
	Lambda = 0
	lastm = 0


	# alpha = float(message_length) / n
	while (float(abs(lastm/n - message_length/n)) > 1/1000.0) and (iterations < 300):
		Lambda = l1 + (l3 - l1) / 2.0
		p_dejoin = getP(Lambda, cost)
		m2 = getHt(p_dejoin)
		if m2 < message_length:
			l3 = Lambda
			m3 = m2
			lastm = m3
		else:
			l1 = Lambda
			lastm = m2
		iterations = iterations + 1
	return Lambda

def cost_hill(X):
	wetCost = 1e10
	hp = np.array([[-0.25, 0.5, -0.25], [0.5, -1, 0.5], [-0.25, 0.5, -0.25]], dtype=np.float32)
	r_1 = signal.convolve2d(X.astype(np.float32), hp, boundary='symm', mode='same')
	lp_1 = np.ones([3, 3], dtype=np.float32)/9
	r_2 = signal.convolve2d(np.abs(r_1), lp_1, boundary='symm', mode='same')
	rho = 1/(r_2+1e-10)
	lp_2 = np.ones([15, 15], dtype=np.float32)/225
	rho = signal.convolve2d(rho, lp_2, boundary='symm', mode='same')
	rho[rho > 50] = wetCost
	rho[np.isnan(rho)] = wetCost
	rho_p = rho.copy()
	rho_m = rho.copy()
	rho_p[X == 255] = wetCost
	rho_m[X == 0] = wetCost
	return rho_p, rho_m

def calCoCostFromInCostFor4(rhoP1, rhoM1):
	cost = np.zeros([81, rhoP1.size // 4], dtype=np.float)
	rho_shape_P = rhoP1.copy()
	rho_shape_M = rhoM1.copy()
	rho_shape_P_1 = rho_shape_P[0: :2, 0: : 2].reshape((1, rhoP1.size // 4))
	rho_shape_P_2 = rho_shape_P[0: :2, 1: : 2].reshape((1, rhoP1.size // 4))
	rho_shape_P_3 = rho_shape_P[1: :2, 0: : 2].reshape((1, rhoP1.size // 4))
	rho_shape_P_4 = rho_shape_P[1: :2, 1: : 2].reshape((1, rhoP1.size // 4))
	rho_shape_M_1 = rho_shape_M[0: :2, 0: : 2].reshape((1, rhoM1.size // 4))
	rho_shape_M_2 = rho_shape_M[0: :2, 1: : 2].reshape((1, rhoM1.size // 4))
	rho_shape_M_3 = rho_shape_M[1: :2, 0: : 2].reshape((1, rhoM1.size // 4))
	rho_shape_M_4 = rho_shape_M[1: :2, 1: : 2].reshape((1, rhoM1.size // 4))
	
	changes = np.array([[1, 1, 1, 1], [1, 1, 1, -1], [1, 1, 1, 0], [1, 1, -1, 1], [1, 1, -1, -1], [1, 1, -1, 0], [1, 1, 0, 1], [1, 1, 0, -1], [1, 1, 0, 0],
						[1, -1, 1, 1], [1, -1, 1, -1], [1, -1, 1, 0], [1, -1, -1, 1], [1, -1, -1, -1], [1, -1, -1, 0], [1, -1, 0, 1], [1, -1, 0, -1], [1, -1, 0, 0],	 
						[1, 0, 1, 1], [1, 0, 1, -1], [1, 0, 1, 0], [1, 0, -1, 1], [1, 0, -1, -1], [1, 0, -1, 0], [1, 0, 0, 1], [1, 0, 0, -1], [1, 0, 0, 0],
						[-1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, 1, 0], [-1, 1, -1, 1], [-1, 1, -1, -1], [-1, 1, -1, 0], [-1, 1, 0, 1], [-1, 1, 0, -1], [-1, 1, 0, 0],
						[-1, -1, 1, 1], [-1, -1, 1, -1], [-1, -1, 1, 0], [-1, -1, -1, 1], [-1, -1, -1, -1], [-1, -1, -1, 0], [-1, -1, 0, 1], [-1, -1, 0, -1], [-1, -1, 0, 0], 
						[-1, 0, 1, 1], [-1, 0, 1, -1], [-1, 0, 1, 0], [-1, 0, -1, 1], [-1, 0, -1, -1], [-1, 0, -1, 0], [-1, 0, 0, 1], [-1, 0, 0, -1], [-1, 0, 0, 0], 
						[0, 1, 1, 1], [0, 1, 1, -1], [0, 1, 1, 0], [0, 1, -1, 1], [0, 1, -1, -1], [0, 1, -1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0],
						[0, -1, 1, 1], [0, -1, 1, -1], [0, -1, 1, 0], [0, -1, -1, 1], [0, -1, -1, -1], [0, -1, -1, 0], [0, -1, 0, 1], [0, -1, 0, -1], [0, -1, 0, 0], 
						[0, 0, 1, 1], [0, 0, 1, -1], [0, 0, 1, 0], [0, 0, -1, 1], [0, 0, -1, -1], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.int)
	weight = np.zeros([1, 81], dtype=np.float)
	
	for i in range(81):
		o = 0
		s = 0
		if ((changes[i, 0] - changes[i, 1] == 0) & changes[i, 0]!=0):
			s = s + 1
		elif abs(changes[i, 0] - changes[i, 1]) == 2:
			o = o + 1
		if ((changes[i, 0] - changes[i, 2] == 0) & changes[i, 0]!=0):
			s = s + 1
		elif abs(changes[i, 0] - changes[i, 2]) == 2:
			o = o + 1
		if ((changes[i, 1] - changes[i, 3] == 0) & changes[i, 3]!=0):
			s = s + 1
		elif abs(changes[i, 1] - changes[i, 3]) == 2:
			o = o + 1
		if ((changes[i, 2] - changes[i, 3] == 0) & changes[i, 3]!=0):
			s = s + 1
		elif abs(changes[i, 2] - changes[i, 3]) == 2:
			o = o + 1
		weight[0, i] = (1+4*o) / (1+s)

	for i in range(81):
		flag = changes[i]
		rho_base = np.zeros([1, rho_shape_M_1.size])
		if flag[0] > 0:
			rho_base = rho_shape_P_1
		elif flag[0] < 0:
			rho_base = rho_shape_M_1
		if flag[1] > 0:
			rho_base = rho_base + rho_shape_P_2
		elif flag[1] < 0:
			rho_base = rho_base + rho_shape_M_2
		if flag[2] > 0:
			rho_base = rho_base + rho_shape_P_3
		elif flag[2] < 0:
			rho_base = rho_base + rho_shape_M_3
		if flag[3] > 0:
			rho_base = rho_base + rho_shape_P_4
		elif flag[3] < 0:
			rho_base = rho_base + rho_shape_M_4
		alpha = weight[0, i]
		cost[i, :] = rho_base * alpha # cost->[81, 256*256/4]
	return cost

def embed(X, joint_rho, m): # X:256, 256  joint_rho: 81, 128, 128
	n = X.size
	X1 = X[0: :2, 0: : 2].reshape(1, -1).copy()
	X2 = X[0: :2, 1: : 2].reshape(1, -1).copy()
	X3 = X[1: :2, 0: : 2].reshape(1, -1).copy()
	X4 = X[1: :2, 1: : 2].reshape(1, -1).copy()
	joint_rho = joint_rho.reshape(81, -1)
	Lambda = calc_lambda(joint_rho, m, n)

	z_dejoin = np.exp(-Lambda * joint_rho)
	z0 = np.zeros([1, joint_rho.shape[1]], dtype=np.float)
	for idx_z in range(81):
		z0 += z_dejoin[idx_z, : ]
	p_dejoin = np.zeros([81, joint_rho.shape[1]], dtype=np.float)
	for idx_z in range(80):
		p_dejoin[idx_z, : ] = z_dejoin[idx_z, : ] / z0
	I = np.ones((1, joint_rho.shape[1]))
	p_dejoin[80, :] = I - np.sum(p_dejoin, axis=0)

	changes = np.array([[1, 1, 1, 1], [1, 1, 1, -1], [1, 1, 1, 0], [1, 1, -1, 1], [1, 1, -1, -1], [1, 1, -1, 0], [1, 1, 0, 1], [1, 1, 0, -1], [1, 1, 0, 0],
						[1, -1, 1, 1], [1, -1, 1, -1], [1, -1, 1, 0], [1, -1, -1, 1], [1, -1, -1, -1], [1, -1, -1, 0], [1, -1, 0, 1], [1, -1, 0, -1], [1, -1, 0, 0],	 
						[1, 0, 1, 1], [1, 0, 1, -1], [1, 0, 1, 0], [1, 0, -1, 1], [1, 0, -1, -1], [1, 0, -1, 0], [1, 0, 0, 1], [1, 0, 0, -1], [1, 0, 0, 0],
						[-1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, 1, 0], [-1, 1, -1, 1], [-1, 1, -1, -1], [-1, 1, -1, 0], [-1, 1, 0, 1], [-1, 1, 0, -1], [-1, 1, 0, 0],
						[-1, -1, 1, 1], [-1, -1, 1, -1], [-1, -1, 1, 0], [-1, -1, -1, 1], [-1, -1, -1, -1], [-1, -1, -1, 0], [-1, -1, 0, 1], [-1, -1, 0, -1], [-1, -1, 0, 0], 
						[-1, 0, 1, 1], [-1, 0, 1, -1], [-1, 0, 1, 0], [-1, 0, -1, 1], [-1, 0, -1, -1], [-1, 0, -1, 0], [-1, 0, 0, 1], [-1, 0, 0, -1], [-1, 0, 0, 0], 
						[0, 1, 1, 1], [0, 1, 1, -1], [0, 1, 1, 0], [0, 1, -1, 1], [0, 1, -1, -1], [0, 1, -1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0],
						[0, -1, 1, 1], [0, -1, 1, -1], [0, -1, 1, 0], [0, -1, -1, 1], [0, -1, -1, -1], [0, -1, -1, 0], [0, -1, 0, 1], [0, -1, 0, -1], [0, -1, 0, 0], 
						[0, 0, 1, 1], [0, 0, 1, -1], [0, 0, 1, 0], [0, 0, -1, 1], [0, 0, -1, -1], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.int)
	randChange = np.random.rand(1, joint_rho.shape[1])
	p_tmp = np.zeros([1, joint_rho.shape[1]], dtype=np.float)

	for idx_v in range(81):
		p_idx = p_tmp + p_dejoin[idx_v, :]
		if changes[idx_v, 0] > 0:
			X1[(randChange < p_idx) & (randChange >= p_tmp) & (X1[ : ] < 255)] += 1
		elif changes[idx_v, 0] < 0:
			X1[(randChange < p_idx) & (randChange >= p_tmp) & (X1[ : ] > 0)] -= 1
		if changes[idx_v, 1] > 0:
			X2[(randChange < p_idx) & (randChange >= p_tmp) & (X2[ : ] < 255)] += 1
		elif changes[idx_v, 1] < 0:
			X2[(randChange < p_idx) & (randChange >= p_tmp) & (X2[ : ] > 0)] -= 1
		if changes[idx_v, 2] > 0:
			X3[(randChange < p_idx) & (randChange >= p_tmp) & (X3[ : ] < 255)] += 1
		elif changes[idx_v, 2] < 0:
			X3[(randChange < p_idx) & (randChange >= p_tmp) & (X3[ : ] > 0)] -= 1
		if changes[idx_v, 3] > 0:
			X4[(randChange < p_idx) & (randChange >= p_tmp) & (X4[ : ] < 255)] += 1
		elif changes[idx_v, 3] < 0:
			X4[(randChange < p_idx) & (randChange >= p_tmp) & (X4[ : ] > 0)] -= 1

		p_tmp = p_idx
	
	stego = np.zeros((X.shape[0], X.shape[1]), dtype=np.uint8)
	stego[0: :2, 0: : 2] = X1.reshape((X.shape[0] // 2, X.shape[0] // 2)).copy()
	stego[0: :2, 1: : 2] = X2.reshape((X.shape[0] // 2, X.shape[0] // 2)).copy()
	stego[1: :2, 0: : 2] = X3.reshape((X.shape[0] // 2, X.shape[0] // 2)).copy()
	stego[1: :2, 1: : 2] = X4.reshape((X.shape[0] // 2, X.shape[0] // 2)).copy()
	return stego

def adjust_cost(rho_ori, grd, factor_adj): # rho_ori (81, 128*128) sign_grd (4, 128, 128)
	# import ipdb
	# ipdb.set_trace()

	sign_grd = np.sign(grd)
	# factor_adj = factor_adj*(abs(grd[0, :]) + abs(grd[1, :]))
	changes = np.array([[1, 1, 1, 1], [1, 1, 1, -1], [1, 1, 1, 0], [1, 1, -1, 1], [1, 1, -1, -1], [1, 1, -1, 0], [1, 1, 0, 1], [1, 1, 0, -1], [1, 1, 0, 0],
						[1, -1, 1, 1], [1, -1, 1, -1], [1, -1, 1, 0], [1, -1, -1, 1], [1, -1, -1, -1], [1, -1, -1, 0], [1, -1, 0, 1], [1, -1, 0, -1], [1, -1, 0, 0],	 
						[1, 0, 1, 1], [1, 0, 1, -1], [1, 0, 1, 0], [1, 0, -1, 1], [1, 0, -1, -1], [1, 0, -1, 0], [1, 0, 0, 1], [1, 0, 0, -1], [1, 0, 0, 0],
						[-1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, 1, 0], [-1, 1, -1, 1], [-1, 1, -1, -1], [-1, 1, -1, 0], [-1, 1, 0, 1], [-1, 1, 0, -1], [-1, 1, 0, 0],
						[-1, -1, 1, 1], [-1, -1, 1, -1], [-1, -1, 1, 0], [-1, -1, -1, 1], [-1, -1, -1, -1], [-1, -1, -1, 0], [-1, -1, 0, 1], [-1, -1, 0, -1], [-1, -1, 0, 0], 
						[-1, 0, 1, 1], [-1, 0, 1, -1], [-1, 0, 1, 0], [-1, 0, -1, 1], [-1, 0, -1, -1], [-1, 0, -1, 0], [-1, 0, 0, 1],
						 [-1, 0, 0, -1], [-1, 0, 0, 0], 
						[0, 1, 1, 1], [0, 1, 1, -1], [0, 1, 1, 0], [0, 1, -1, 1], [0, 1, -1, -1], [0, 1, -1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0],
						[0, -1, 1, 1], [0, -1, 1, -1], [0, -1, 1, 0], [0, -1, -1, 1], [0, -1, -1, -1], [0, -1, -1, 0], [0, -1, 0, 1], [0, -1, 0, -1], [0, -1, 0, 0], 
						[0, 0, 1, 1], [0, 0, 1, -1], [0, 0, 1, 0], [0, 0, -1, 1], [0, 0, -1, -1], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.int)
	rho_ret = rho_ori.copy()
	for idx_v in range(80):
		grd_l = np.zeros([sign_grd.shape[1], sign_grd.shape[2]], dtype=np.int)
		grd_l[(sign_grd[0, :, :] == changes[idx_v, 0]) & (sign_grd[1, :, :] == changes[idx_v, 1]) & \
				(sign_grd[2, :, :] == changes[idx_v, 2])] = 1
		grd_l[(sign_grd[0, :, :] == changes[idx_v, 0]) & (sign_grd[1, :, :] == changes[idx_v, 1]) & \
				(sign_grd[3, :, :] == changes[idx_v, 3])] = 1
		grd_l[(sign_grd[0, :, :] == changes[idx_v, 0]) & (sign_grd[2, :, :] == changes[idx_v, 2]) & \
				(sign_grd[3, :, :] == changes[idx_v, 3])] = 1
		grd_l[(sign_grd[1, :, :] == changes[idx_v, 1]) & (sign_grd[2, :, :] == changes[idx_v, 2]) & \
				(sign_grd[3, :, :] == changes[idx_v, 3])] = 1
	
		grd_l[(sign_grd[0, :, :] == -changes[idx_v, 0]) & (sign_grd[1, :, :] == -changes[idx_v, 1]) & \
				(sign_grd[2, :, :] == -changes[idx_v, 2])] = -1
		grd_l[(sign_grd[0, :, :] == -changes[idx_v, 0]) & (sign_grd[1, :, :] == -changes[idx_v, 1]) & \
				(sign_grd[3, :, :] == -changes[idx_v, 3])] = -1
		grd_l[(sign_grd[0, :, :] == -changes[idx_v, 0]) & (sign_grd[2, :, :] == -changes[idx_v, 2]) & \
				(sign_grd[3, :, :] == -changes[idx_v, 3])] = -1
		grd_l[(sign_grd[1, :, :] == -changes[idx_v, 1]) & (sign_grd[2, :, :] == -changes[idx_v, 2]) & \
				(sign_grd[3, :, :] == -changes[idx_v, 3])] = -1
				
		rho_i = rho_ret[idx_v, :, :]
		rho_i[grd_l == 1] /= factor_adj
		rho_i[grd_l == -1] *= factor_adj
		
		# for i in range(4):
		# 	rho_i[sign_grd[i, :, :] == changes[idx_v, i]] /= factor_adj
		# 	rho_i[sign_grd[i, :, :] == -changes[idx_v, i]] *= factor_adj
		rho_ret[idx_v, :, :] = rho_i
	return rho_ret

def jointrho(cover):
	rhop, rhom = cost_hill(cover)
	cost = calCoCostFromInCostFor4(rhop, rhom)
	return cost

def grad_adjust_cost(rho_ori, grd, factor_adj): # rho_ori (9, 256*128) sign_grd (2, 256*128)
	# import ipdb
	# ipdb.set_trace()

	sign_grd = np.sign(grd)
	factor_adj = factor_adj*(1 + 10*(abs(grd[0, :]) + abs(grd[1, :])))
	changes = np.array([[1, 1], [1, -1], [1, 0], 
						[-1, 1], [-1, -1], [-1, 0], 
						[0, 1], [0, -1], [0, 0]], dtype=np.int)
	rho_ret = rho_ori.copy()
	for idx_v in range(8):
		grd_l = np.zeros([sign_grd.shape[1]], dtype=np.int)
		grd_l[(sign_grd[0, :] == changes[idx_v, 0]) & (sign_grd[1, :] == changes[idx_v, 1])] = 1
		grd_l[(sign_grd[0, :] == -changes[idx_v, 0]) & (sign_grd[1, :] == -changes[idx_v, 1])] = -1  	
		rho_i = rho_ret[idx_v, :]
		rho_i[grd_l == 1] /= factor_adj[grd_l == 1]
		rho_i[grd_l == -1] *= factor_adj[grd_l == -1]
		rho_ret[idx_v, :] = rho_i
	return rho_ret

