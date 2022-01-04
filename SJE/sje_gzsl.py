import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, log_loss, f1_score

parser = argparse.ArgumentParser(description="GZSL with SJE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-mr', '--margin', default=1, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)
parser.add_argument('-acc', '--accuracy', help='choose between top1, top5, logloss, F1, all', default='all', type=str) # UPDATE #####################

class SJE():
	
	def __init__(self, args):

		self.args = args

		random.seed(self.args.rand_seed)
		np.random.seed(self.args.rand_seed)

		data_folder = '../xlsa17/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		trainval_loc = 'trainval_loc'
		test_seen_loc = 'test_seen_loc'
		test_unseen_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_trainval_gzsl = feat[:, np.squeeze(att_splits[trainval_loc]-1)]
		self.X_test_seen = feat[:, np.squeeze(att_splits[test_seen_loc]-1)]
		self.X_test_unseen = feat[:, np.squeeze(att_splits[test_unseen_loc]-1)]

		labels = res101['labels']
		self.labels_trainval_gzsl = np.squeeze(labels[np.squeeze(att_splits[trainval_loc]-1)])
		self.labels_test_seen = np.squeeze(labels[np.squeeze(att_splits[test_seen_loc]-1)])
		self.labels_test_unseen = np.squeeze(labels[np.squeeze(att_splits[test_unseen_loc]-1)])
		self.labels_test = np.concatenate((self.labels_test_seen, self.labels_test_unseen), axis=0)

		train_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)]))
		val_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)]))
		trainval_classes_seen = np.unique(self.labels_trainval_gzsl)
		self.test_classes_seen = np.unique(self.labels_test_seen)
		self.test_classes_unseen = np.unique(self.labels_test_unseen)
		test_classes = np.unique(self.labels_test) # All Classes of the dataset

		train_gzsl_indices=[]
		val_gzsl_indices=[]

		for cl in train_classes:
			train_gzsl_indices = train_gzsl_indices + np.squeeze(np.where(self.labels_trainval_gzsl==cl)).tolist()

		for cl in val_classes:
			val_gzsl_indices = val_gzsl_indices + np.squeeze(np.where(self.labels_trainval_gzsl==cl)).tolist()

		train_gzsl_indices = sorted(train_gzsl_indices)
		val_gzsl_indices = sorted(val_gzsl_indices)
		
		self.X_train_gzsl = self.X_trainval_gzsl[:, np.array(train_gzsl_indices)]
		self.labels_train_gzsl = self.labels_trainval_gzsl[np.array(train_gzsl_indices)]
		
		self.X_val_gzsl = self.X_trainval_gzsl[:, np.array(val_gzsl_indices)]
		self.labels_val_gzsl = self.labels_trainval_gzsl[np.array(val_gzsl_indices)]

		print('Tr:{}; Val:{}; Tr+Val:{}; Test Seen:{}; Test Unseen:{}\n'.format(self.X_train_gzsl.shape[1], self.X_val_gzsl.shape[1], 
			                                                                    self.X_trainval_gzsl.shape[1], self.X_test_seen.shape[1], 
			                                                                    self.X_test_unseen.shape[1]))

		i=0
		for labels in trainval_classes_seen:
			self.labels_trainval_gzsl[self.labels_trainval_gzsl == labels] = i    
			i+=1

		j=0
		for labels in train_classes:
			self.labels_train_gzsl[self.labels_train_gzsl == labels] = j
			j+=1

		k=0
		for labels in val_classes:
			self.labels_val_gzsl[self.labels_val_gzsl == labels] = k
			k+=1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.trainval_sig = sig[:, trainval_classes_seen-1]
		self.train_sig = sig[:, train_classes-1]
		self.val_sig = sig[:, val_classes-1]
		self.test_sig = sig[:, test_classes-1] # Entire Signature Matrix

		if self.args.accuracy=='all':
			if self.args.dataset=='CUB':
				scaler_train = preprocessing.StandardScaler()
				scaler_trainval = preprocessing.StandardScaler()
				
				scaler_train.fit(self.X_train_gzsl.T)
				scaler_trainval.fit(self.X_trainval_gzsl.T)

				self.X_train_gzsl = scaler_train.transform(self.X_train_gzsl.T).T
				self.X_val_gzsl = scaler_train.transform(self.X_val_gzsl.T).T
				
				self.X_trainval_gzsl = scaler_trainval.transform(self.X_trainval_gzsl.T).T
				self.X_test_seen = scaler_trainval.transform(self.X_test_seen.T).T
				self.X_test_unseen = scaler_trainval.transform(self.X_test_unseen.T).T
			if self.args.dataset=='AWA1':
				self.X_train_gzsl = self.normalizeFeature(self.X_train_gzsl.T).T
				self.X_trainval_gzsl = self.normalizeFeature(self.X_trainval_gzsl.T).T
			if self.args.dataset=='AWA2':
				self.X_train_gzsl = self.normalizeFeature(self.X_train_gzsl.T).T
				self.X_trainval_gzsl = self.normalizeFeature(self.X_trainval_gzsl.T).T
			if self.args.dataset=='SUN':
				scaler_train = preprocessing.StandardScaler()
				scaler_trainval = preprocessing.StandardScaler()
				
				scaler_train.fit(self.X_train_gzsl.T)
				scaler_trainval.fit(self.X_trainval_gzsl.T)

				self.X_train_gzsl = scaler_train.transform(self.X_train_gzsl.T).T
				self.X_val_gzsl = scaler_train.transform(self.X_val_gzsl.T).T
				
				self.X_trainval_gzsl = scaler_trainval.transform(self.X_trainval_gzsl.T).T
				self.X_test_seen = scaler_trainval.transform(self.X_test_seen.T).T
				self.X_test_unseen = scaler_trainval.transform(self.X_test_unseen.T).T
		else:
			if self.args.norm_type=='std':
				scaler_train = preprocessing.StandardScaler()
				scaler_trainval = preprocessing.StandardScaler()
				
				scaler_train.fit(self.X_train_gzsl.T)
				scaler_trainval.fit(self.X_trainval_gzsl.T)

				self.X_train_gzsl = scaler_train.transform(self.X_train_gzsl.T).T
				self.X_val_gzsl = scaler_train.transform(self.X_val_gzsl.T).T
				
				self.X_trainval_gzsl = scaler_trainval.transform(self.X_trainval_gzsl.T).T
				self.X_test_seen = scaler_trainval.transform(self.X_test_seen.T).T
				self.X_test_unseen = scaler_trainval.transform(self.X_test_unseen.T).T

			if self.args.norm_type=='L2':
				self.X_train_gzsl = self.normalizeFeature(self.X_train_gzsl.T).T
				self.X_trainval_gzsl = self.normalizeFeature(self.X_trainval_gzsl.T).T
				# self.X_val = self.normalizeFeature(self.X_val.T).T
				# self.X_test = self.normalizeFeature(self.X_test.T).T

	def normalizeFeature(self, x):
	    # x = N x d (d:feature dimension, N:number of instances)
		x = x + 1e-10
		feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
		feat = x / feature_norm[:, np.newaxis]

		return feat

	def find_compatible_y(self, X_n, W, y_n, sig):
		if self.args.accuracy=='all':
			if self.args.dataset=='CUB':
				margin = 4.0
			if self.args.dataset=='AWA1':
				margin = 2.5
			if self.args.dataset=='AWA2':
				margin = 2.5
			if self.args.dataset=='APY':
				margin = 1.5
			if self.args.dataset=='SUN':
				margin = 2.0
		else:
			margin = self.args.margin

		XW = np.dot(X_n, W)
		# Scale the projected vector
		XW = preprocessing.scale(XW)
		scores = np.zeros(sig.shape[1])
		scores[y_n] = 0.0
		gt_class_score = np.dot(XW, sig[:, y_n])
		
		for i in range(sig.shape[1]):
			if i!=y_n:
				scores[i] = margin + np.dot(XW, sig[:, i]) - gt_class_score

		return np.argmax(scores)

	def update_W(self, X, labels, sig, W, idx):
		if self.args.accuracy=='all':
			if self.args.dataset=='CUB':
				lr = 0.1
			if self.args.dataset=='AWA1':
				lr = 1.0
			if self.args.dataset=='AWA2':
				lr = 1.0
			if self.args.dataset=='APY':
				lr = 0.01
			if self.args.dataset=='SUN':
				lr = 1.0
		else:
			lr = self.args.lr
		
		for j in idx:
			X_n = X[:, j]
			y_n = labels[j]
			y = self.find_compatible_y(X_n, W, y_n, sig)
			
			if y!=y_n:
				Y = np.expand_dims(sig[:, y_n]-sig[:, y], axis=0)
				W += lr*np.dot(np.expand_dims(X_n, axis=1), Y)
		
		return W

	def fit_train(self):

		print('Training on train set...\n')

		########################################## UPDATE #################################################
		if args.accuracy=='logloss':
			best_val_acc = 100000
			best_tr_acc = 100000
		else:
			best_val_acc = 0.0
			best_tr_acc = 0.0
		########################################## UPDATE #################################################
		# best_val_acc = 0.0
		# best_tr_acc = 0.0
		best_val_ep = -1
		best_tr_ep = -1
		
		rand_idx = np.arange(self.X_train_gzsl.shape[1])

		W = np.random.rand(self.X_train_gzsl.shape[0], self.train_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		for ep in range(self.args.epochs):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(self.X_train_gzsl, self.labels_train_gzsl, self.train_sig, W, rand_idx)
			
			########################################## UPDATE #################################################
			if args.accuracy=='top1':
				val_acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				tr_acc = self.zsl_acc(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)
			if args.accuracy=='top5':
				val_acc = self.zsl_acc_top5(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				tr_acc = self.zsl_acc_top5(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)
			if args.accuracy=='logloss':
				val_acc = self.zsl_acc_logloss(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				tr_acc = self.zsl_acc_logloss(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)
			if args.accuracy=='F1':
				val_acc = self.zsl_acc_f1(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				tr_acc = self.zsl_acc_f1(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)
			if args.accuracy=='all':
				val_acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				tr_acc = self.zsl_acc(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)
			########################################## UPDATE #################################################
			# val_acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
			# tr_acc = self.zsl_acc(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, val_acc, elapsed//60, elapsed%60))
			
			########################################## UPDATE #################################################
			if args.accuracy=='logloss':
				if val_acc<best_val_acc:
					best_val_acc = val_acc
					best_val_ep = ep+1
				
				if tr_acc<best_tr_acc:
					best_tr_ep = ep+1
					best_tr_acc = tr_acc
			else:
				if val_acc>best_val_acc:
					best_val_acc = val_acc
					best_val_ep = ep+1
				
				if tr_acc>best_tr_acc:
					best_tr_ep = ep+1
					best_tr_acc = tr_acc
			########################################## UPDATE #################################################
			# if val_acc>best_val_acc:
				# best_val_acc = val_acc
				# best_val_ep = ep+1
			
			# if tr_acc>best_tr_acc:
				# best_tr_ep = ep+1
				# best_tr_acc = tr_acc

			if ep+1-best_val_ep>self.args.early_stop:
				print('Early Stopping by {} epochs. Exiting...'.format(self.args.epochs-(ep+1)))
				break

		print('Best Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
		
		return best_val_ep

	def fit_trainval(self):

		print('\nTraining on trainval set for GZSL...\n')

		best_tr_acc = 0.0
		best_tr_ep = -1
		
		rand_idx = np.arange(self.X_trainval_gzsl.shape[1])

		W = np.random.rand(self.X_trainval_gzsl.shape[0], self.trainval_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		for ep in range(self.num_epochs_trainval):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(self.X_trainval_gzsl, self.labels_trainval_gzsl, self.trainval_sig, W, rand_idx)
			
			########################################## UPDATE #################################################
			if args.accuracy=='top1':
				tr_acc = self.zsl_acc(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)
			if args.accuracy=='top5':
				tr_acc = self.zsl_acc_top5(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)
			if args.accuracy=='logloss':
				tr_acc = self.zsl_acc_logloss(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)
			if args.accuracy=='F1':
				tr_acc = self.zsl_acc_f1(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)
			if args.accuracy=='all':
				tr_acc = self.zsl_acc(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)
			########################################## UPDATE #################################################
			tr_acc = self.zsl_acc(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Trainval Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, elapsed//60, elapsed%60))
						
			if tr_acc>best_tr_acc:
				best_tr_ep = ep+1
				best_tr_acc = tr_acc
				best_W = np.copy(W)

		print('Best Trainval Acc:{} @ Epoch {}\n'.format(best_tr_acc, best_tr_ep))
		
		return best_W

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.array([np.argmax(output) for output in dist])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		return acc

	def zsl_acc_top5(self, X, W, y_true, sig): # Class Averaged Top-5 Accuarcy
		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.argpartition(dist, kth=-1, axis=-1)[:,-5:]
		correct_pred = 0
		for i in range(len(y_true)):
			if y_true[i] in predicted_classes[i]:
				correct_pred += 1
		acc = correct_pred / len(y_true)

		return acc

	def zsl_acc_logloss(self, X, W, y_true, sig): # Class Averaged LogLoss Accuarcy
		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		acc = log_loss(y_true, dist)

		return acc

	def zsl_acc_f1(self, X, W, y_true, sig): # Class Averaged F1 Score Accuarcy
		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.array([np.argmax(output) for output in dist])
		acc = f1_score( y_true, predicted_classes, average='micro')

		return acc

	def zsl_acc_gzsl(self, X, W, y_true, classes, sig, is_seen): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		y_pred = np.array([np.argmax(output)+1 for output in class_scores])

		########################################## UPDATE #################################################
		if is_seen:
			a_file = open("testing/gzsl/sje_dist_seen_"+self.args.dataset+".txt", "w")
			b_file = open("testing/gzsl/sje_pred_seen_"+self.args.dataset+".txt", "w")
			np.savetxt(a_file, class_scores)
			np.savetxt(b_file, y_pred)
			a_file.close()
			b_file.close()
		if not is_seen:
			a_file = open("testing/gzsl/sje_dist_unseen_"+self.args.dataset+".txt", "w")
			b_file = open("testing/gzsl/sje_pred_unseen_"+self.args.dataset+".txt", "w")
			np.savetxt(a_file, class_scores)
			np.savetxt(b_file, y_pred)
			a_file.close()
			b_file.close()
		########################################## UPDATE #################################################

		per_class_acc = np.zeros(len(classes))

		for i in range(len(classes)):
			is_class = y_true==classes[i]
			per_class_acc[i] = ((y_pred[is_class]==y_true[is_class]).sum())/is_class.sum()
		
		return per_class_acc.mean()

	def zsl_acc_gzsl_top5(self, X, W, y_true, classes, sig): # Class Averaged Top-5 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		y_pred = np.argsort(class_scores)[:,-5:]
		y_pred = y_pred + 1

		per_class_acc = np.zeros(len(classes))

		for i in range(len(classes)):
			is_class = y_true==classes[i]
			per_class_acc[i] = ((y_pred[is_class]==classes[i]).sum())/is_class.sum()
		
		return per_class_acc.mean()

	def zsl_acc_gzsl_logloss(self, X, W, y_true, classes, sig): # Class Averaged LogLoss Accuarcy
		
		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		acc = log_loss(y_true, class_scores, labels=classes)

		return acc

	def zsl_acc_gzsl_f1(self, X, W, y_true, classes, sig): # Class Averaged LogLoss Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		y_pred = np.array([np.argmax(output)+1 for output in class_scores])
		acc = f1_score(y_true, y_pred, average='micro', labels=classes)

		return acc

	def evaluate(self):

		self.num_epochs_trainval = self.fit_train()

		best_W = self.fit_trainval()

		print('Testing...\n')

		########################################## UPDATE #################################################
		if args.accuracy=='top1':
			acc_seen_classes = self.zsl_acc_gzsl(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig, True)
			acc_unseen_classes = self.zsl_acc_gzsl(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig, False)
		if args.accuracy=='top5':
			acc_seen_classes = self.zsl_acc_gzsl_top5(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig)
			acc_unseen_classes = self.zsl_acc_gzsl_top5(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig)
		if args.accuracy=='logloss':
			acc_seen_classes = self.zsl_acc_gzsl_logloss(self.X_test_seen, best_W, self.labels_test_seen, np.unique(self.labels_test), self.test_sig)
			acc_unseen_classes = self.zsl_acc_gzsl_logloss(self.X_test_unseen, best_W, self.labels_test_unseen, np.unique(self.labels_test), self.test_sig)
		if args.accuracy=='F1':
			acc_seen_classes = self.zsl_acc_gzsl_f1(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig)
			acc_unseen_classes = self.zsl_acc_gzsl_f1(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig)
		if args.accuracy=='all':
			acctop1_seen_classes = self.zsl_acc_gzsl(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig, True)
			acctop1_unseen_classes = self.zsl_acc_gzsl(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig, False)
			acctop5_seen_classes = self.zsl_acc_gzsl_top5(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig)
			acctop5_unseen_classes = self.zsl_acc_gzsl_top5(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig)
			acclogloss_seen_classes = self.zsl_acc_gzsl_logloss(self.X_test_seen, best_W, self.labels_test_seen, np.unique(self.labels_test), self.test_sig)
			acclogloss_unseen_classes = self.zsl_acc_gzsl_logloss(self.X_test_unseen, best_W, self.labels_test_unseen, np.unique(self.labels_test), self.test_sig)
			accf1_seen_classes = self.zsl_acc_gzsl_f1(self.X_test_seen, best_W, self.labels_test_seen, np.unique(self.labels_test), self.test_sig)
			accf1_unseen_classes = self.zsl_acc_gzsl_f1(self.X_test_unseen, best_W, self.labels_test_unseen, np.unique(self.labels_test), self.test_sig)
		########################################## UPDATE #################################################
		# acc_seen_classes = self.zsl_acc_gzsl(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig)
		# acc_unseen_classes = self.zsl_acc_gzsl(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig)

		if args.accuracy=='all':
			HM_top1 = 2*acctop1_seen_classes*acctop1_unseen_classes/(acctop1_seen_classes+acctop1_unseen_classes)
			HM_top5 = 2*acctop5_seen_classes*acctop5_unseen_classes/(acctop5_seen_classes+acctop5_unseen_classes)
			HM_logloss = 2*acclogloss_seen_classes*acclogloss_unseen_classes/(acclogloss_seen_classes+acclogloss_unseen_classes)
			HM_f1 = 2*accf1_seen_classes*accf1_unseen_classes/(accf1_seen_classes+accf1_unseen_classes)

			print('Top 1 U:{}; S:{}; H:{}'.format(acctop1_unseen_classes, acctop1_seen_classes, HM_top1))
			print('Top 5 U:{}; S:{}; H:{}'.format(acctop5_unseen_classes, acctop5_seen_classes, HM_top5))
			print('LogLoss U:{}; S:{}; H:{}'.format(acclogloss_unseen_classes, acclogloss_seen_classes, HM_logloss))
			print('F1 U:{}; S:{}; H:{}'.format(accf1_unseen_classes, accf1_seen_classes, HM_f1))
		else:
			HM = 2*acc_seen_classes*acc_unseen_classes/(acc_seen_classes+acc_unseen_classes)

			print('U:{}; S:{}; H:{}'.format(acc_unseen_classes, acc_seen_classes, HM))

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = SJE(args)	
	clf.evaluate()
