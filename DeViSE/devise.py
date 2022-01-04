import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, log_loss, f1_score

parser = argparse.ArgumentParser(description="DeViSE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-mr', '--margin', default=1, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)
parser.add_argument('-acc', '--accuracy', help='choose between top1, top5, logloss, F1, all', default='all', type=str) # UPDATE #####################

"""

Best Values of (norm, lr, mr) found by validation & corr. test accuracies:

SUN  -> (None, 0.01, 3.0)-> Test Acc : 0.5569
CUB  -> (L2, 1.0, 1.0)   -> Test Acc : 0.4407
APY  -> (L2, 1.0, 1.0)   -> Test Acc : 0.3333
AWA1 -> (std, 0.01, 200) -> Test Acc : 0.5525
AWA2 -> (std, 0.001, 150)-> Test Acc : 0.5768

"""

class DeViSE():
	
	def __init__(self, args):

		self.args = args

		random.seed(self.args.rand_seed)
		np.random.seed(self.args.rand_seed)

		data_folder = '../xlsa17/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
		self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		self.labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
		self.labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
		self.labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])

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

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

		if self.args.accuracy=='all':
			if self.args.dataset=='CUB':
				print('L2 norm Scaling\n')
				self.X_train = self.normalizeFeature(self.X_train.T).T
			if self.args.dataset=='AWA1':
				print('Standard Scaling\n')
				scaler = preprocessing.StandardScaler()
				scaler.fit(self.X_train.T)

				self.X_train = scaler.transform(self.X_train.T).T
				self.X_val = scaler.transform(self.X_val.T).T
				self.X_test = scaler.transform(self.X_test.T).T
			if self.args.dataset=='AWA2':
				print('Standard Scaling\n')
				scaler = preprocessing.StandardScaler()
				scaler.fit(self.X_train.T)

				self.X_train = scaler.transform(self.X_train.T).T
				self.X_val = scaler.transform(self.X_val.T).T
				self.X_test = scaler.transform(self.X_test.T).T
			if self.args.dataset=='APY':
				print('L2 norm Scaling\n')
				self.X_train = self.normalizeFeature(self.X_train.T).T
		else:
			if self.args.norm_type=='std':
				print('Standard Scaling\n')
				scaler = preprocessing.StandardScaler()
				scaler.fit(self.X_train.T)

				self.X_train = scaler.transform(self.X_train.T).T
				self.X_val = scaler.transform(self.X_val.T).T
				self.X_test = scaler.transform(self.X_test.T).T

			if self.args.norm_type=='L2':
				print('L2 norm Scaling\n')
				self.X_train = self.normalizeFeature(self.X_train.T).T
				# self.X_val = self.normalizeFeature(self.X_val.T).T
				# self.X_test = self.normalizeFeature(self.X_test.T).T

	def normalizeFeature(self, x):
	    # x = N x d (d:feature dimension, N:number of instances)
		x = x + 1e-10
		feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
		feat = x / feature_norm[:, np.newaxis]

		return feat

	def update_W(self, W, idx, train_classes):
		if self.args.accuracy=='all':
			if self.args.dataset=='CUB':
				lr = 1.0
				margin = 1.0
			if self.args.dataset=='AWA1':
				lr = 0.01
				margin = 200
			if self.args.dataset=='AWA2':
				lr = 0.001
				margin = 150
			if self.args.dataset=='APY':
				lr = 1.0
				margin = 1.0
			if self.args.dataset=='SUN':
				lr = 0.01
				margin = 3.0
		else:
			lr = self.args.lr
			margin = self.args.margin
		
		for j in idx:
			
			X_n = self.X_train[:, j]
			y_n = self.labels_train[j]
			y_ = train_classes[train_classes!=y_n]
			XW = np.dot(X_n, W)
			gt_class_score = np.dot(XW, self.train_sig[:, y_n])

			for i, label in enumerate(y_):
				score = margin+np.dot(XW, self.train_sig[:, label])-gt_class_score
				if score>0:
					Y = np.expand_dims(self.train_sig[:, y_n]-self.train_sig[:, label], axis=0)
					W += lr*np.dot(np.expand_dims(X_n, axis=1), Y)
					break # acc. to the authors, it was practical to stop after first margin violating term was found
		return W

	def fit(self):

		print('Training...\n')

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
		
		rand_idx = np.arange(self.X_train.shape[1])

		W = np.random.rand(self.X_train.shape[0], self.train_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		train_classes = np.unique(self.labels_train)

		for ep in range(self.args.epochs):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(W, rand_idx, train_classes)

			########################################## UPDATE #################################################
			if args.accuracy=='top1':
				tr_acc = self.zsl_acc(self.X_train, W, self.labels_train, self.train_sig)			
				val_acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
			if args.accuracy=='top5':
				tr_acc = self.zsl_acc_top5(self.X_train, W, self.labels_train, self.train_sig)			
				val_acc = self.zsl_acc_top5(self.X_val, W, self.labels_val, self.val_sig)
			if args.accuracy=='logloss':
				tr_acc = self.zsl_acc_logloss(self.X_train, W, self.labels_train, self.train_sig)			
				val_acc = self.zsl_acc_logloss(self.X_val, W, self.labels_val, self.val_sig)
			if args.accuracy=='F1':
				tr_acc = self.zsl_acc_f1(self.X_train, W, self.labels_train, self.train_sig)			
				val_acc = self.zsl_acc_f1(self.X_val, W, self.labels_val, self.val_sig)
			if args.accuracy=='all':
				tr_acc = self.zsl_acc(self.X_train, W, self.labels_train, self.train_sig)			
				val_acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
			########################################## UPDATE #################################################
			# tr_acc = self.zsl_acc(self.X_train, W, self.labels_train, self.train_sig)			
			# val_acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, val_acc, elapsed//60, elapsed%60))
			
			########################################## UPDATE #################################################
			if args.accuracy=='logloss':
				if val_acc<best_val_acc:
					best_val_acc = val_acc
					best_val_ep = ep+1
					best_W = np.copy(W)
				
				if tr_acc<best_tr_acc:
					best_tr_ep = ep+1
					best_tr_acc = tr_acc
			else:
				if val_acc>best_val_acc:
					best_val_acc = val_acc
					best_val_ep = ep+1
					best_W = np.copy(W)
				
				if tr_acc>best_tr_acc:
					best_tr_ep = ep+1
					best_tr_acc = tr_acc
			########################################## UPDATE #################################################
			# if val_acc>best_val_acc:
				# best_val_acc = val_acc
				# best_val_ep = ep+1
				# best_W = np.copy(W)
			
			# if tr_acc>best_tr_acc:
				# best_tr_ep = ep+1
				# best_tr_acc = tr_acc

			if ep+1-best_val_ep>self.args.early_stop:
				print('Early Stopping by {} epochs. Exiting...'.format(self.args.epochs-(ep+1)))
				break

		print('\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
		
		return best_W

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.array([np.argmax(output) for output in dist])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		########################################## UPDATE #################################################
		correct_pred = []
		for i in range(len(y_true)):
			correct_pred.append(1) if y_true[i] == predicted_classes[i] else correct_pred.append(0)
		a_file = open("testing/zsl/devise_dist_"+self.args.dataset+".txt", "w")
		b_file = open("testing/zsl/devise_pred_"+self.args.dataset+".txt", "w")
		c_file = open("testing/zsl/devise_"+self.args.dataset+".txt", "w")
		np.savetxt(a_file, dist)
		np.savetxt(b_file, predicted_classes)
		np.savetxt(c_file, correct_pred)
		a_file.close()
		b_file.close()
		c_file.close()
		########################################## UPDATE #################################################

		return acc

	def zsl_acc_top5(self, X, W, y_true, sig): # Class Averaged Top-5 Accuarcy
		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.argpartition(dist, kth=-1, axis=-1)[:,-5:]
		classes = np.unique(y_true)
		acc = 0
		for i in range(len(classes)):
			correct_predictions = 0
			samples = 0
			for j in range(len(y_true)):
				if y_true[j] == classes[i]:
					samples += 1
					if y_true[j] in predicted_classes[j]:
						correct_predictions += 1
			if samples == 0:
				acc += 1
			else:
				acc += correct_predictions/samples

		acc = acc/len(classes)

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

	def evaluate(self):

		best_W = self.fit()

		print('Testing...\n')

		########################################## UPDATE #################################################
		if args.accuracy=='top1':
			test_acc = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)
		if args.accuracy=='top5':
			test_acc = self.zsl_acc_top5(self.X_test, best_W, self.labels_test, self.test_sig)
		if args.accuracy=='logloss':
			test_acc = self.zsl_acc_logloss(self.X_test, best_W, self.labels_test, self.test_sig)
		if args.accuracy=='F1':
			test_acc = self.zsl_acc_f1(self.X_test, best_W, self.labels_test, self.test_sig)
		if args.accuracy=='all':
			test_acc_top_1 = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)
			test_acc_top_5 = self.zsl_acc_top5(self.X_test, best_W, self.labels_test, self.test_sig)
			test_acc_logloss = self.zsl_acc_logloss(self.X_test, best_W, self.labels_test, self.test_sig)
			test_acc_f1 = self.zsl_acc_f1(self.X_test, best_W, self.labels_test, self.test_sig)
		########################################## UPDATE #################################################
		# test_acc = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)

		if args.accuracy=='all':
			print('Test Acc top1:{}; top5:{}; logloss:{}; f1:{};'.format(test_acc_top_1,test_acc_top_5,test_acc_logloss,test_acc_f1))
		else:
			print('Test Acc:{}'.format(test_acc))

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = DeViSE(args)	
	clf.evaluate()
