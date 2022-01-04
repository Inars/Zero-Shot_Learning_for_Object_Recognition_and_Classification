import numpy as np
import argparse
from scipy import io, spatial, linalg
from sklearn.metrics import confusion_matrix, log_loss, f1_score

parser = argparse.ArgumentParser(description="SAE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-mode', '--mode', help='train/test, if test set alpha, gamma to best values below', default='train', type=str)
parser.add_argument('-ld1', '--ld1', default=5, help='best value for F-->S during test, lower bound of variation interval during train', type=float)
parser.add_argument('-ld2', '--ld2', default=5, help='best value for S-->F during test, upper bound of variation interval during train', type=float)
parser.add_argument('-acc', '--accuracy', help='choose between top1, top5, logloss, F1, all', default='all', type=str) # UPDATE #####################

"""

Range of Lambda for Validation:

AWA1 -> 2-8 for [F-->S] and 0.4-1.6 for [S-->F]
AWA2 -> 0.1-1.6
CUB  -> 50-5000 for [F-->S] and 0.05-5 for [S-->F]
SUN  -> 0.005-5
APY  -> 0.5-50

Best Value of Lambda found by validation & corr. test accuracies:
		   				
AWA1 -> 0.5134 @ 3.0  [F-->S] 0.5989 @ 0.8  [S-->F]
AWA2 -> 0.5166 @ 0.6  [F-->S] 0.6051 @ 0.2  [S-->F]
CUB  -> 0.3948 @ 100  [F-->S] 0.4670 @ 0.2  [S-->F]
SUN  -> 0.5285 @ 0.32 [F-->S] 0.5986 @ 0.16 [S-->F]
APY  -> 0.1607 @ 2.0  [F-->S] 0.1650 @ 4.0  [S-->F] 

"""
class SAE():
	
	def __init__(self, args):

		self.args = args

		data_folder = '../xlsa17/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		feat = res101['features']
		self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
		self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		self.labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
		self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
		self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)]

		train_labels_seen = np.unique(self.labels_train)
		val_labels_unseen = np.unique(self.labels_val)
		test_labels_unseen = np.unique(self.labels_test)

		i=0
		for labels in train_labels_seen:
			self.labels_train[self.labels_train == labels] = i    
			i+=1
		
		j=0
		for labels in val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j+=1
		
		k=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = k
			k+=1

		sig = att_splits['att']# k x C
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

		self.train_att = np.zeros((self.X_train.shape[1], self.train_sig.shape[0]))
		for i in range(self.train_att.shape[0]):
			self.train_att[i] = self.train_sig.T[self.labels_train[i][0]]

		self.X_train = self.normalizeFeature(self.X_train.T).T

	def normalizeFeature(self, x):
		# x = N x d (d:feature dimension, N:number of instances)
		x = x + 1e-10
		feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
		feat = x / feature_norm[:, np.newaxis]

		return feat

	def find_W(self, X, S, ld):

		# INPUTS:
	    # X: d x N - data matrix
	    # S: Number of Attributes (k) x N - semantic matrix
	    # ld: regularization parameter
	    #
	    # Return :
	    # 	W: kxd projection matrix

		A = np.dot(S, S.T)
		B = ld*np.dot(X, X.T)
		C = (1+ld)*np.dot(S, X.T)
		W = linalg.solve_sylvester(A, B, C)

		return W

	def find_lambda(self):

		print('Training...\n')

		########################################## UPDATE #################################################
		if args.accuracy=='logloss':
			best_acc_F2S = 100000
			best_acc_S2F = 100000
		else:
			best_acc_F2S = 0.0
			best_acc_S2F = 0.0
		########################################## UPDATE #################################################
		# best_acc_F2S = 0.0
		# best_acc_S2F = 0.0

		if args.accuracy=='all':
			if args.dataset=='CUB':
				lambda_F2S = 100
				lambda_S2F = 0.2

				ld = 100
			if args.dataset=='AWA1':
				lambda_F2S = 3.0
				lambda_S2F = 0.8

				ld = 3.0
			if args.dataset=='AWA2':
				lambda_F2S = 0.6
				lambda_S2F = 0.2

				ld = 0.6
			if args.dataset=='APY':
				lambda_F2S = 2.0
				lambda_S2F = 4.0

				ld = 2.0
			if args.dataset=='SUN':
				lambda_F2S = 0.32
				lambda_S2F = 0.16

				ld = 0.32
		else:
			lambda_F2S = self.args.ld1
			lambda_S2F = self.args.ld2

			ld = self.args.ld1
		
		
		########################################## UPDATE #################################################
		W = self.find_W(self.X_train, self.train_att.T, ld)

		if args.accuracy=='top1':
			acc_F2S, acc_S2F = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig, 'val')
		if args.accuracy=='top5':
			acc_F2S, acc_S2F = self.zsl_acc_top5(self.X_val, W, self.labels_val, self.val_sig, 'val')
		if args.accuracy=='logloss':
			acc_F2S, acc_S2F = self.zsl_acc_logloss(self.X_val, W, self.labels_val, self.val_sig, 'val')
		if args.accuracy=='F1':
			acc_F2S, acc_S2F = self.zsl_acc_f1(self.X_val, W, self.labels_val, self.val_sig, 'val')
		if args.accuracy=='all':
			acc_F2S, acc_S2F = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig, 'val')

		print('Val Acc --> [F-->S]:{} [S-->F]:{} @ lambda = {}\n'.format(acc_F2S, acc_S2F, ld))

		if args.accuracy=='logloss':
			if acc_F2S<best_acc_F2S:
				best_acc_F2S = acc_F2S
				lambda_F2S = ld
				best_W_F2S = np.copy(W)

			if acc_S2F<best_acc_S2F:
				best_acc_S2F = acc_S2F
				lambda_S2F = ld
				best_W_S2F = np.copy(W)
		else:
			if acc_F2S>best_acc_F2S:
				best_acc_F2S = acc_F2S
				lambda_F2S = ld
				best_W_F2S = np.copy(W)

			if acc_S2F>best_acc_S2F:
				best_acc_S2F = acc_S2F
				lambda_S2F = ld
				best_W_S2F = np.copy(W)
			
		ld*=2
		########################################## UPDATE #################################################

		while (ld<=self.args.ld2):
			
			W = self.find_W(self.X_train, self.train_att.T, ld)

			########################################## UPDATE #################################################
			if args.accuracy=='top1':
				acc_F2S, acc_S2F = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig, 'val')
			if args.accuracy=='top5':
				acc_F2S, acc_S2F = self.zsl_acc_top5(self.X_val, W, self.labels_val, self.val_sig, 'val')
			if args.accuracy=='logloss':
				acc_F2S, acc_S2F = self.zsl_acc_logloss(self.X_val, W, self.labels_val, self.val_sig, 'val')
			if args.accuracy=='F1':
				acc_F2S, acc_S2F = self.zsl_acc_f1(self.X_val, W, self.labels_val, self.val_sig, 'val')
			if args.accuracy=='all':
				acc_F2S, acc_S2F = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig, 'val')
			########################################## UPDATE #################################################
			# acc_F2S, acc_S2F = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig, 'val')

			print('Val Acc --> [F-->S]:{} [S-->F]:{} @ lambda = {}\n'.format(acc_F2S, acc_S2F, ld))

			########################################## UPDATE #################################################
			if args.accuracy=='logloss':
				if acc_F2S<best_acc_F2S:
					best_acc_F2S = acc_F2S
					lambda_F2S = ld
					best_W_F2S = np.copy(W)

				if acc_S2F<best_acc_S2F:
					best_acc_S2F = acc_S2F
					lambda_S2F = ld
					best_W_S2F = np.copy(W)
			else:
				if acc_F2S>best_acc_F2S:
					best_acc_F2S = acc_F2S
					lambda_F2S = ld
					best_W_F2S = np.copy(W)

				if acc_S2F>best_acc_S2F:
					best_acc_S2F = acc_S2F
					lambda_S2F = ld
					best_W_S2F = np.copy(W)
			########################################## UPDATE #################################################
			# if acc_F2S>best_acc_F2S:
				# best_acc_F2S = acc_F2S
				# lambda_F2S = ld
				# best_W_F2S = np.copy(W)

			# if acc_S2F>best_acc_S2F:
				# best_acc_S2F = acc_S2F
				# lambda_S2F = ld
				# best_W_S2F = np.copy(W)
			
			ld*=2

		print('\nBest Val Acc --> [F-->S]:{} @ lambda = {} [S-->F]:{} @ lambda = {}\n'.format(best_acc_F2S, lambda_F2S, best_acc_S2F, lambda_S2F))
		
		return best_W_F2S, best_W_S2F

	def zsl_acc(self, X, W, y_true, sig, mode='val'): # Class Averaged Top-1 Accuarcy

		if mode=='F2S':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
			cm_F2S = confusion_matrix(y_true, pred_F2S)
			cm_F2S = cm_F2S.astype('float')/cm_F2S.sum(axis=1)[:, np.newaxis]
			acc_F2S = sum(cm_F2S.diagonal())/sig.shape[1]

			########################################## UPDATE #################################################
			correct_pred = []
			for i in range(len(y_true)):
				correct_pred.append(1) if y_true[i] == pred_F2S[i] else correct_pred.append(0)
			a_file = open("testing/zsl/sae_dist_"+self.args.dataset+".txt", "w")
			b_file = open("testing/zsl/sae_pred_"+self.args.dataset+".txt", "w")
			c_file = open("testing/zsl/sae_"+self.args.dataset+".txt", "w")
			np.savetxt(a_file, dist_F2S)
			np.savetxt(b_file, pred_F2S)
			np.savetxt(c_file, correct_pred)
			a_file.close()
			b_file.close()
			c_file.close()
			########################################## UPDATE #################################################

			return acc_F2S

		if mode=='S2F':
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
			cm_S2F = confusion_matrix(y_true, pred_S2F)
			cm_S2F = cm_S2F.astype('float')/cm_S2F.sum(axis=1)[:, np.newaxis]
			acc_S2F = sum(cm_S2F.diagonal())/sig.shape[1]

			return acc_S2F		

		if mode=='val':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			
			pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
			pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
			
			cm_F2S = confusion_matrix(y_true, pred_F2S)
			cm_F2S = cm_F2S.astype('float')/cm_F2S.sum(axis=1)[:, np.newaxis]

			cm_S2F = confusion_matrix(y_true, pred_S2F)
			cm_S2F = cm_S2F.astype('float')/cm_S2F.sum(axis=1)[:, np.newaxis]
			
			acc_F2S = sum(cm_F2S.diagonal())/sig.shape[1]
			acc_S2F = sum(cm_S2F.diagonal())/sig.shape[1]

			# acc = acc_F2S if acc_F2S>acc_S2F else acc_S2F

			return acc_F2S, acc_S2F

	def zsl_acc_top5(self, X, W, y_true, sig, mode='val'): # Class Averaged Top-5 Accuarcy

		if mode=='F2S':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			pred_F2S = np.argpartition(dist_F2S, kth=-1, axis=-1)[:,-5:]
			classes = np.unique(y_true)
			acc_F2S = 0
			for i in range(len(classes)):
				correct_predictions = 0
				samples = 0
				for j in range(len(y_true)):
					if y_true[j] == classes[i]:
						samples += 1
						if y_true[j] in pred_F2S[j]:
							correct_predictions += 1
				if samples == 0:
					acc_F2S += 1
				else:
					acc_F2S += correct_predictions/samples

			acc_F2S = acc_F2S/len(classes)

			return acc_F2S

		if mode=='S2F':
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			pred_S2F = np.argpartition(dist_S2F, kth=-1, axis=-1)[:,-5:]
			classes = np.unique(y_true)
			acc_S2F = 0
			for i in range(len(classes)):
				correct_predictions = 0
				samples = 0
				for j in range(len(y_true)):
					if y_true[j] == classes[i]:
						samples += 1
						if y_true[j] in pred_S2F[j]:
							correct_predictions += 1
				if samples == 0:
					acc_S2F += 1
				else:
					acc_S2F += correct_predictions/samples

			acc_S2F = acc_S2F/len(classes)

			return acc_S2F	

		if mode=='val':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			
			pred_F2S = np.argpartition(dist_F2S, kth=-1, axis=-1)[:,-5:]
			pred_S2F = np.argpartition(dist_S2F, kth=-1, axis=-1)[:,-5:]
			
			classes = np.unique(y_true)

			acc_F2S = 0
			for i in range(len(classes)):
				correct_predictions = 0
				samples = 0
				for j in range(len(y_true)):
					if y_true[j] == classes[i]:
						samples += 1
						if y_true[j] in pred_F2S[j]:
							correct_predictions += 1
				if samples == 0:
					acc_F2S += 1
				else:
					acc_F2S += correct_predictions/samples

			acc_S2F = 0
			for i in range(len(classes)):
				correct_predictions = 0
				samples = 0
				for j in range(len(y_true)):
					if y_true[j] == classes[i]:
						samples += 1
						if y_true[j] in pred_S2F[j]:
							correct_predictions += 1
				if samples == 0:
					acc_S2F += 1
				else:
					acc_S2F += correct_predictions/samples

			acc_F2S = acc_F2S/len(classes)
			acc_S2F = acc_S2F/len(classes)

			# acc = acc_F2S if acc_F2S>acc_S2F else acc_S2F

			return acc_F2S, acc_S2F

	def zsl_acc_logloss(self, X, W, y_true, sig, mode='val'): # Class Averaged LogLoss Accuarcy

		if mode=='F2S':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			acc_F2S = log_loss(y_true, dist_F2S)

			return acc_F2S

		if mode=='S2F':
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			acc_S2F = log_loss(y_true, dist_S2F)

			return acc_S2F		

		if mode=='val':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			
			acc_F2S = log_loss(y_true, dist_F2S)
			acc_S2F = log_loss(y_true, dist_S2F)

			# acc = acc_F2S if acc_F2S>acc_S2F else acc_S2F

			return acc_F2S, acc_S2F

	def zsl_acc_f1(self, X, W, y_true, sig, mode='val'): # Class Averaged F1 Score Accuarcy

		if mode=='F2S':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
			acc_F2S = f1_score( y_true, pred_F2S, average='micro')

			return acc_F2S

		if mode=='S2F':
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
			acc_S2F = f1_score( y_true, pred_S2F, average='micro')

			return acc_S2F		

		if mode=='val':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))
			dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
			
			pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
			pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
			
			acc_F2S = f1_score( y_true, pred_F2S, average='micro')
			acc_S2F = f1_score( y_true, pred_S2F, average='micro')

			# acc = acc_F2S if acc_F2S>acc_S2F else acc_S2F

			return acc_F2S, acc_S2F

	def evaluate(self):

		if self.args.mode=='train': best_W_F2S, best_W_S2F = self.find_lambda()
		else: 
			if args.accuracy=='all':
				if args.dataset=='CUB':
					self.args.ld1 = 100
					self.args.ld2 = 0.2
				if args.dataset=='AWA1':
					self.args.ld1 = 3.0
					self.args.ld2 = 0.8
				if args.dataset=='AWA2':
					self.args.ld1 = 0.6
					self.args.ld2 = 0.2
				if args.dataset=='APY':
					self.args.ld1 = 2.0
					self.args.ld2 = 4.0
				if args.dataset=='SUN':
					self.args.ld1 = 0.32
					self.args.ld2 = 0.16
			best_W_F2S = self.find_W(self.X_train, self.train_att.T, self.args.ld1)
			best_W_S2F = self.find_W(self.X_train, self.train_att.T, self.args.ld2)

		########################################## UPDATE #################################################
		if args.accuracy=='top1':
			test_acc_F2S = self.zsl_acc(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_S2F = self.zsl_acc(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
		if args.accuracy=='top5':
			test_acc_F2S = self.zsl_acc_top5(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_S2F = self.zsl_acc_top5(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
		if args.accuracy=='logloss':
			test_acc_F2S = self.zsl_acc_logloss(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_S2F = self.zsl_acc_logloss(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
		if args.accuracy=='F1':
			test_acc_F2S = self.zsl_acc_f1(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_S2F = self.zsl_acc_f1(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
		if args.accuracy=='all':
			test_acc_top1_F2S = self.zsl_acc(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_top1_S2F = self.zsl_acc(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
			test_acc_top5_F2S = self.zsl_acc_top5(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_top5_S2F = self.zsl_acc_top5(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
			test_acc_logloss_F2S = self.zsl_acc_logloss(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_logloss_S2F = self.zsl_acc_logloss(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
			test_acc_f1_F2S = self.zsl_acc_f1(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
			test_acc_f1_S2F = self.zsl_acc_f1(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')
		########################################## UPDATE #################################################
		# test_acc_F2S = self.zsl_acc(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
		# test_acc_S2F = self.zsl_acc(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')

		if args.accuracy=='all':
			print('Test Acc Top 1 --> [F-->S]:{} [S-->F]:{}'.format(test_acc_top1_F2S, test_acc_top1_S2F))
			print('Test Acc Top 5 --> [F-->S]:{} [S-->F]:{}'.format(test_acc_top5_F2S, test_acc_top5_S2F))
			print('Test Acc LogLoss --> [F-->S]:{} [S-->F]:{}'.format(test_acc_logloss_F2S, test_acc_logloss_S2F))
			print('Test Acc F1 --> [F-->S]:{} [S-->F]:{}'.format(test_acc_f1_F2S, test_acc_f1_S2F))
		else:
			print('Test Acc --> [F-->S]:{} [S-->F]:{}'.format(test_acc_F2S, test_acc_S2F))

if __name__ == '__main__':

	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	clf = SAE(args)
	clf.evaluate()
