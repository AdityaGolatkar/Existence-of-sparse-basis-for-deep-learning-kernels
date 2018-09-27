# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:59:47 2018

@author: Rudrajit Das
"""

# x=[x1,x2,..,x10], y = 1 if (sum(xi^4)/10)^0.5 > theta else 0
# architecture used 10*100*100*1 NN

import numpy as np
from sklearn.linear_model import ElasticNet
import scipy.io as sio


#####################################
# x1 = input vector 1
# x2 = input vector 2
# gamma = parameter of the RBF kernel
# k = value of the kernel function
#####################################
def kernel(x1,x2,gamma):   
	#k = np.exp(-gamma*(np.linalg.norm(x1-x2))**2)
	k = np.dot(x1,x2)
	return k

###############################################
# X = data matrix with each point along the row
# gamma = parameter of the RBF kernel
# G = non centered Gram Matrix
# G_mean_shifted = Centred Gram Matrix
###############################################

def gram_generate(X,gamma):
	no_of_points = X.shape[0]
	print(no_of_points)
	G = np.zeros((no_of_points,no_of_points))
	for i in range(no_of_points):
		for j in range(no_of_points):
			G[i,j] =  kernel(X[i,:],X[j,:],gamma)
            
	N = G.shape[0]
	G_row = np.sum(G,0)/N
	G_sum = np.sum(G_row)/N    
	G_mean_shifted = np.zeros((no_of_points,no_of_points)) 
	for i in range(no_of_points):
		for j in range(no_of_points):
			G_mean_shifted[i,j] = G[i,j] - G_row[i] - G_row[j] + G_sum
    
	return G, G_mean_shifted


#################################################
# G = Centred Gram Matrix
# eigen_value = Eigen Value of the Gram Matrix
# eigen_vecktor = Eigen Vector of the Gram Matrix
#################################################

def gram_eigen_vectors(G,num_eig_vecs):
	eigen_value, eigen_vector = np.linalg.eigh(G)
	idx = eigen_value.argsort()[::-1] 
	eigen_value = eigen_value[idx]   
	eigen_vector = eigen_vector[:,idx]   
	for i in range(num_eig_vecs):
			eigen_vector[:,i] = (eigen_vector[:,i]/np.sqrt(eigen_value[i]))
            
	return eigen_value[0:num_eig_vecs],eigen_vector[:,0:num_eig_vecs]

#######################################
# Data = in this case is the Gram Matrix
# no_of_ev_of_gram = is the number of 
# eigen vectors of the Gram Matrx 
# returned matrix = contain the Sparse
# Eigen Vectors in columns
#######################################

def naive_spca(data,no_of_ev_of_gram,r):

	no_of_points = data.shape[0]
	feature_size = data.shape[1]
	sparse_pc_old = np.zeros((feature_size,no_of_ev_of_gram))
	################################
	#SVD gives u s v.T and not u s v
	################################
	u,s,v = np.linalg.svd(data,full_matrices=0)
	#print(s)
	s2 = np.diag(s)    
	for i in range (s2.shape[0]):
		s2[i,i] = 1/np.sqrt(s[i])
	#print(v[0:5,:].T)
	#print(np.dot(v[0:5,:],v[0:5,:].T))    
	
	j = 0
	sparse_pc_list = []
	sparse_pc_list.append(sparse_pc_old)
	
	while(j < 6):
		sparse_pc = np.zeros((feature_size,no_of_ev_of_gram))
		if j == 0:
			a = v[0:no_of_ev_of_gram,0:].T
		j=j+1
		for i in range(no_of_ev_of_gram):
			y = data.dot(a[0:,i])
			#y = np.dot(data,a[0:,i])
			elastic = ElasticNet(alpha=2, l1_ratio=r, max_iter=10000)
			elastic.fit(data*np.sqrt(2*no_of_points),y*np.sqrt(2*no_of_points))
			pc = elastic.coef_
			#print(pc)
			sparse_pc[0:,i] = pc
		u1,s1,v1=np.linalg.svd(np.dot(np.dot(np.dot(s2,u.T),np.dot(data.T,data)),sparse_pc),full_matrices=0)
		#u1,s1,v1=np.linalg.svd(np.dot(np.dot(data.T,data),sparse_pc),full_matrices=0)
		#pdb.set_trace()
		#a = u1[0:,0:sig_dim].dot(v1)
		a = np.dot(np.dot(u,s2),u1.dot(v1))
		#a = u1.dot(v1)
		#print(sparse_pc)
		if ((np.linalg.norm(sparse_pc-sparse_pc_list[j-1],ord='fro')))<0.0008:
			#print((np.linalg.norm(sparse_pc-sparse_pc_list[j-1],ord='fro')))
			sparse_pc_list.append(sparse_pc)	
			break
		sparse_pc_list.append(sparse_pc)
	#print(sparse_pc_list)
	nrm = np.sqrt(np.sum(sparse_pc_list[len(sparse_pc_list)-1]*sparse_pc_list[len(sparse_pc_list)-1],axis=0))
	#print(nrm)
	#sparse_pc_list[len(sparse_pc_list)-1]=sparse_pc_list[len(sparse_pc_list)-1]/nrm
	for i in range(no_of_ev_of_gram): 
		sc = np.sqrt(np.dot(sparse_pc_list[len(sparse_pc_list)-1][:,i].T,np.dot(data,sparse_pc_list[len(sparse_pc_list)-1][:,i])))           
		sparse_pc_list[len(sparse_pc_list)-1][:,i] = (1/sc)*sparse_pc_list[len(sparse_pc_list)-1][:,i]		
		#sparse_pc_list[len(sparse_pc_list)-1][:,i] = (1/np.sqrt(s[i]))*sparse_pc_list[len(sparse_pc_list)-1][:,i]
	return sparse_pc_list[len(sparse_pc_list)-1]


#####################################################
# alphas = contains the eigen vectors of G in columns
# test_point should be a row vector
# G_not_cen = Gram Matrix
# test_point = testing point
# gamma = parameter of the kernel
# returned value = reconstruction error
######################################################

def recon_error(test_point,data,alphas,sum_alpha,alpha_G_row,G_sum,gamma):
	n = data.shape[0]
	k = np.zeros(n)
	for j in range(n):
		k[j] = kernel(test_point,data[j,:],gamma)

	f = np.dot(k,alphas) - np.dot(sum_alpha,(np.sum(k)/n)-G_sum) - alpha_G_row

	err = kernel(test_point,test_point,gamma) - 2*(np.sum(k)/n) + G_sum - f.T.dot(f)
    
	return err    


def main():
    
	#points_per_digit = 400
	#testpoints_per_samedigit = 450   
	no_of_ev_of_gram = 15

	z_train_full = np.squeeze(np.load('Z_train.npy'))
	z_train = z_train_full[0:1000,:]
	labels = sio.loadmat('y_train.mat')
	labels_train_full = labels['y_train']
	labels_train = labels_train_full[0:1000]
	#labes = labels_train[labels_train==0]
	#print(labes.shape)
	one_idx = np.nonzero(labels_train)
	print(z_train.shape)
	#print(one_idx.shape)
	data = z_train[one_idx[0],:]
	print(data.shape)
	points_per_digit = data.shape[0]
	threshold = 0.9

	sigma = 0            
	for i in range(points_per_digit):
		for j in range(points_per_digit):
			sigma =  sigma + (np.linalg.norm(data[i,:]-data[j,:]))**2
	gamma = 1/(2*sigma/(points_per_digit*points_per_digit))

	G_not_cen, G_cen = gram_generate(data,gamma)
	#sio.savemat('G_cen.mat', {'G_cen':G_cen})    
	actual_e_val,actual_e_vec = gram_eigen_vectors(G_cen,no_of_ev_of_gram)
	#print(actual_e_val[0:10])  
	l1_ratio = 2.5
	sparse_e_vec = naive_spca(G_cen,no_of_ev_of_gram,l1_ratio)
	#np.save('sparse.npy',sparse_e_vec)
	#sparse_e_vec = np.load('sparse.npy')
    
	#alphas = actual_e_vec[:,0:no_of_ev_of_gram]#sparse_e_vec
	alphas = sparse_e_vec
	alpha_sum = np.sum(alphas,0)
	G_row = np.sum(G_not_cen,0)/points_per_digit
	G_sum = np.sum(G_row)/points_per_digit
	alpha_G_row = np.dot(G_row,alphas)

	z_test = np.squeeze(np.load('Z_test.npy'))
	labels2 = sio.loadmat('y_test.mat')
	labels_test = labels2['y_test']
	one_idx2 = np.nonzero(labels_test)
	test_data_same = z_test[one_idx2[0], :]
	test_data_diff = z_test[np.where(labels_test==0)[0], :]

    
	correct_same = 0
	err_same = np.zeros(test_data_same.shape[0])

	for i in range(test_data_same.shape[0]):
		
		error = recon_error(test_data_same[i,0:],data,alphas,alpha_sum,alpha_G_row,G_sum,gamma)
		err_same[i] = error
		if (abs(error)) <= threshold:
			correct_same = correct_same + 1
		#print(error)            
		#print(i+1)
		#print(correct_same)
  
	correct_diff = 0
	err_diff = np.zeros(test_data_diff.shape[0])

	for i in range(test_data_diff.shape[0]):
		
		error = recon_error(test_data_diff[i,0:],data,alphas,alpha_sum,alpha_G_row,G_sum,gamma)
		err_diff[i] = error
		if (abs(error)) >= threshold:
			correct_diff = correct_diff + 1
		#print(error)            
		#print(i+1)
		#print(correct_diff)

	percent_same_correct = correct_same/test_data_same.shape[0]
	print("Same Accuracy Sparse :",percent_same_correct) 
	#print("Same Error :",avg_err_same/test_data_same.shape[0]) 
	# Avg. same error with 0 PCs = 36.018   

	percent_diff_correct = correct_diff/test_data_diff.shape[0]
	print("Diff Accuracy Sparse :",percent_diff_correct)
	#print("Diff Error :",avg_err_diff/test_data_diff.shape[0])  
	# Avg. diff error with 0 PCs = 78.632
    
	percent_correct = (correct_same+correct_diff)/(test_data_same.shape[0]+test_data_diff.shape[0])
	print("Overall Accuracy Sparse :",percent_correct)

	tot_ct = 0        
	for i in range(no_of_ev_of_gram):          
		ct = 0		 
		for j in range(points_per_digit):		                 
			if abs(sparse_e_vec[j,i]) > 0.000001:
			     ct=ct+1
		print(ct)
		tot_ct = tot_ct+ct 
     
	for i in range(no_of_ev_of_gram):          
		ct = 0		 
		for j in range(points_per_digit):		                 
			if abs(actual_e_vec[j,i]) > 0.000001:
			     ct=ct+1
		print(ct) 
        
	tot_ct = int(tot_ct/no_of_ev_of_gram) 
	print(tot_ct)


	alphas = actual_e_vec[:,0:no_of_ev_of_gram]#sparse_e_vec
	#alphas = sparse_e_vec
	alpha_sum = np.sum(alphas,0)
	G_row = np.sum(G_not_cen,0)/points_per_digit
	G_sum = np.sum(G_row)/points_per_digit
	alpha_G_row = np.dot(G_row,alphas)

	z_test = np.squeeze(np.load('Z_test.npy'))
	labels2 = sio.loadmat('y_test.mat')
	labels_test = labels2['y_test']
	one_idx2 = np.nonzero(labels_test)
	test_data_same = z_test[one_idx2[0], :]
	test_data_diff = z_test[np.where(labels_test==0)[0], :]

    
	correct_same = 0
	err_same2 = np.zeros(test_data_same.shape[0])

	for i in range(test_data_same.shape[0]):
		
		error = recon_error(test_data_same[i,0:],data,alphas,alpha_sum,alpha_G_row,G_sum,gamma)
		err_same2[i] = error
		if (abs(error)) <= threshold:
			correct_same = correct_same + 1
		#print(error)            
		#print(i+1)
		#print(correct_same)
  
	correct_diff = 0
	err_diff2 = np.zeros(test_data_diff.shape[0])

	for i in range(test_data_diff.shape[0]):
		
		error = recon_error(test_data_diff[i,0:],data,alphas,alpha_sum,alpha_G_row,G_sum,gamma)
		err_diff2[i] = error
		if (abs(error)) >= threshold:
			correct_diff = correct_diff + 1
		#print(error)            
		#print(i+1)
		#print(correct_diff)

	percent_same_correct = correct_same/test_data_same.shape[0]
	print("Same Accuracy Original :",percent_same_correct) 
	#print("Same Error :",avg_err_same/test_data_same.shape[0]) 
	# Avg. same error with 0 PCs = 36.018   

	percent_diff_correct = correct_diff/test_data_diff.shape[0]
	print("Diff Accuracy Original :",percent_diff_correct)
	#print("Diff Error :",avg_err_diff/test_data_diff.shape[0])  
	# Avg. diff error with 0 PCs = 78.632
    
	percent_correct = (correct_same+correct_diff)/(test_data_same.shape[0]+test_data_diff.shape[0])
	print("Overall Accuracy Original :",percent_correct)

	RMSE_same = np.linalg.norm(err_same2-err_same)/np.linalg.norm(err_same2)
	print("RMSE same: ",RMSE_same)

	RMSE_diff = np.linalg.norm(err_diff2-err_diff)/np.linalg.norm(err_diff2)
	print("RMSE diff: ",RMSE_diff)

	
main()