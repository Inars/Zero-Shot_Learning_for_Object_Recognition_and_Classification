import numpy as np
import argparse
from scipy import io
from sklearn.metrics import confusion_matrix, log_loss, f1_score

parser = argparse.ArgumentParser(description="ESZSL")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-mode', '--mode', help='train/test, if test set alpha, gamma to best values below', default='train', type=str)
parser.add_argument('-alpha', '--alpha', default=0, type=int)
parser.add_argument('-gamma', '--gamma', default=0, type=int)
parser.add_argument('-acc', '--accuracy', help='choose between top1, top5, logloss, F1, all', default='all', type=str) # UPDATE #####################

"""

Alpha --> Regularizer for Kernel/Feature Space
Gamma --> Regularizer for Attribute Space

Best Values of (Alpha, Gamma) found by validation & corr. test accuracies:

AWA1 -> (3, 0)  -> Test Acc : 0.5680
AWA2 -> (3, 0)  -> Test Acc : 0.5482
CUB  -> (3, -1) -> Test Acc : 0.5394
SUN  -> (3, 2)  -> Test Acc : 0.5569
APY  -> (3, -1) -> Test Acc : 0.3856

"""

class ESZSL():
	
	def __init__(self, args):

		self.args = args

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
		self.X_trainval = np.concatenate((self.X_train, self.X_val), axis=1)
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
		self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
		labels_trainval = np.concatenate((labels_train, self.labels_val), axis=0)
		self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)]

		train_labels_seen = np.unique(labels_train)
		val_labels_unseen = np.unique(self.labels_val)
		trainval_labels_seen = np.unique(labels_trainval)
		test_labels_unseen = np.unique(self.labels_test)

		i=0
		for labels in train_labels_seen:
			labels_train[labels_train == labels] = i    
			i+=1
		
		j=0
		for labels in val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j+=1
		
		k=0
		for labels in trainval_labels_seen:
			labels_trainval[labels_trainval == labels] = k
			k+=1
		
		l=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = l
			l+=1

		self.gt_train = np.zeros((labels_train.shape[0], len(train_labels_seen)))
		self.gt_train[np.arange(labels_train.shape[0]), np.squeeze(labels_train)] = 1

		self.gt_trainval = np.zeros((labels_trainval.shape[0], len(trainval_labels_seen)))
		self.gt_trainval[np.arange(labels_trainval.shape[0]), np.squeeze(labels_trainval)] = 1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.trainval_sig = sig[:, trainval_labels_seen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

	def find_W(self, X, y, sig, alpha, gamma):

		part_0 = np.linalg.pinv(np.matmul(X, X.T) + (10**alpha)*np.eye(X.shape[0]))
		part_1 = np.matmul(np.matmul(X, y), sig.T)
		part_2 = np.linalg.pinv(np.matmul(sig, sig.T) + (10**gamma)*np.eye(sig.shape[0]))

		W = np.matmul(np.matmul(part_0, part_1), part_2) # Feature Dimension x Number of Attributes

		return W

	def fit(self):

		print('Training...\n')

		########################################## UPDATE #################################################
		if args.accuracy=='logloss':
			best_acc = 100000
		else:
			best_acc = 0.0
		########################################## UPDATE #################################################
		# best_acc = 0.0

		for alph in range(-3, 4):
			for gamm in range(-3, 4):
				W = self.find_W(self.X_train, self.gt_train, self.train_sig, alph, gamm)
				########################################## UPDATE #################################################
				if args.accuracy=='top1':		
					acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
				if args.accuracy=='top5':
					acc = self.zsl_acc_top5(self.X_val, W, self.labels_val, self.val_sig)
				if args.accuracy=='logloss':
					acc = self.zsl_acc_logloss(self.X_val, W, self.labels_val, self.val_sig)
				if args.accuracy=='F1':
					acc = self.zsl_acc_f1(self.X_val, W, self.labels_val, self.val_sig)
				if args.accuracy=='all':		
					acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
				########################################## UPDATE #################################################
				# acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
				print('Val Acc:{}; Alpha:{}; Gamma:{}\n'.format(acc, alph, gamm))
				########################################## UPDATE #################################################
				if args.accuracy=='logloss':
					if acc<best_acc:
						best_acc = acc
						alpha = alph
						gamma = gamm
				else:
					if acc>best_acc:
						best_acc = acc
						alpha = alph
						gamma = gamm
				########################################## UPDATE #################################################
				# if acc>best_acc:
					# best_acc = acc
					# alpha = alph
					# gamma = gamm

		print('\nBest Val Acc:{} with Alpha:{} & Gamma:{}\n'.format(best_acc, alpha, gamma))
		
		return alpha, gamma

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		predicted_classes = np.array([np.argmax(output) for output in class_scores])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		########################################## UPDATE #################################################
		correct_pred = []
		for i in range(len(y_true)):
			correct_pred.append(1) if y_true[i] == predicted_classes[i] else correct_pred.append(0)
		a_file = open("testing/zsl/eszsl_dist_"+self.args.dataset+".txt", "w")
		b_file = open("testing/zsl/eszsl_pred_"+self.args.dataset+".txt", "w")
		c_file = open("testing/zsl/eszsl_"+self.args.dataset+".txt", "w")
		np.savetxt(a_file, class_scores)
		np.savetxt(b_file, predicted_classes)
		np.savetxt(c_file, correct_pred)
		a_file.close()
		b_file.close()
		c_file.close()
		########################################## UPDATE #################################################

		return acc

	def zsl_acc_top5(self, X, W, y_true, sig): # Class Averaged Top-5 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		predicted_classes = np.argpartition(class_scores, kth=-1, axis=-1)[:,-5:]
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

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		acc = log_loss(y_true, class_scores)

		return acc

	def zsl_acc_f1(self, X, W, y_true, sig): # Class Averaged F1 Score Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		predicted_classes = np.array([np.argmax(output) for output in class_scores])
		acc = f1_score( y_true, predicted_classes, average='micro')

		return acc

	# def zsl_acc(self, X, W, y_true, classes, sig): # Class Averaged Top-1 Accuarcy

	# 	class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
	# 	y_pred = np.array([np.argmax(output) for output in class_scores])
	# 	y_true = np.squeeze(y_true)

	# 	per_class_acc = np.zeros(len(classes))

	# 	for i in range(len(classes)):
	# 		is_class = y_true==classes[i]
	# 		per_class_acc[i] = ((y_pred[is_class]==y_true[is_class]).sum())/is_class.sum()
		
	# 	return per_class_acc.mean()

	def evaluate(self, alpha, gamma):

		print('Testing...\n')

		best_W = self.find_W(self.X_trainval, self.gt_trainval, self.trainval_sig, alpha, gamma) # combine train and val

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
	
	clf = ESZSL(args)
	
	if args.mode=='train': 
		args.alpha, args.gamma = clf.fit()
	else:
		if args.dataset=='CUB':
			args.alpha, args.gamma = 3, -1
		if args.dataset=='AWA1':
			args.alpha, args.gamma = 3, 0
		if args.dataset=='AWA2':
			args.alpha, args.gamma = 3, 0
		if args.dataset=='APY':
			args.alpha, args.gamma = 3, -1
		if args.dataset=='SUN':
			args.alpha, args.gamma = 3, 2
	
	clf.evaluate(args.alpha, args.gamma)
