import numpy as np
import argparse
from scipy import io
from sklearn.metrics import confusion_matrix, log_loss, f1_score

parser = argparse.ArgumentParser(description="GZSL with ESZSL")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-mode', '--mode', help='train/test, if test, set alpha, gamma to best values as given below', default='train', type=str)
parser.add_argument('-alpha', '--alpha', default=0, type=int)
parser.add_argument('-gamma', '--gamma', default=0, type=int)
parser.add_argument('-acc', '--accuracy', help='choose between top1, top5, logloss, F1, all', default='all', type=str) # UPDATE #####################

"""

Alpha --> Regularizer for Kernel/Feature Space
Gamma --> Regularizer for Attribute Space

Best Values of (Alpha, Gamma) found by validation & corr. test accuracies:

AWA1 -> (3, 0) -> Seen : 0.8684 Unseen : 0.0529 HM : 0.0998
AWA2 -> (3, 0) -> Seen : 0.8884 Unseen : 0.0404 HM : 0.0772
CUB  -> (3, 0) -> Seen : 0.5653 Unseen : 0.1470 HM : 0.2334
SUN  -> (3, 2) -> Seen : 0.2841 Unseen : 0.1375 HM : 0.1853
APY  -> (2, 0) -> Seen : 0.8107 Unseen : 0.0225 HM : 0.0439

"""

class ESZSL():
	
	def __init__(self, args):

		self.args = args

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

		# Train and Val are first separated to find the best hyperparamters on val and then to finally use them to train on trainval set.

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

		self.gt_train_gzsl = np.zeros((self.labels_train_gzsl.shape[0], len(train_classes)))
		self.gt_train_gzsl[np.arange(self.labels_train_gzsl.shape[0]), self.labels_train_gzsl] = 1

		self.gt_trainval = np.zeros((self.labels_trainval_gzsl.shape[0], len(trainval_classes_seen)))
		self.gt_trainval[np.arange(self.labels_trainval_gzsl.shape[0]), self.labels_trainval_gzsl] = 1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.trainval_sig = sig[:, trainval_classes_seen-1]
		self.train_sig = sig[:, train_classes-1]
		self.val_sig = sig[:, val_classes-1]
		self.test_sig = sig[:, test_classes-1] # Entire Signature Matrix

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
				W = self.find_W(self.X_train_gzsl, self.gt_train_gzsl, self.train_sig, alph, gamm)
				
				########################################## UPDATE #################################################
				if args.accuracy=='top1':
					acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				if args.accuracy=='top5':
					acc = self.zsl_acc_top5(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				if args.accuracy=='logloss':
					acc = self.zsl_acc_logloss(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				if args.accuracy=='F1':
					acc = self.zsl_acc_f1(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				if args.accuracy=='all':
					acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				########################################## UPDATE #################################################
				# acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)

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

		return acc

	def zsl_acc_top5(self, X, W, y_true, sig): # Class Averaged Top-5 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		predicted_classes = np.argpartition(class_scores, kth=-1, axis=-1)[:,-5:]
		correct_pred = 0
		for i in range(len(y_true)):
			if y_true[i] in predicted_classes[i]:
				correct_pred += 1
		acc = correct_pred / len(y_true)

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

	def zsl_acc_gzsl(self, X, W, y_true, classes, sig, is_seen): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		y_pred = np.array([np.argmax(output)+1 for output in class_scores])

		########################################## UPDATE #################################################
		if is_seen:
			a_file = open("testing/gzsl/eszsl_dist_seen_"+self.args.dataset+".txt", "w")
			b_file = open("testing/gzsl/eszsl_pred_seen_"+self.args.dataset+".txt", "w")
			np.savetxt(a_file, class_scores)
			np.savetxt(b_file, y_pred)
			a_file.close()
			b_file.close()
		if not is_seen:
			a_file = open("testing/gzsl/eszsl_dist_unseen_"+self.args.dataset+".txt", "w")
			b_file = open("testing/gzsl/eszsl_pred_unseen_"+self.args.dataset+".txt", "w")
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

	def evaluate(self, alpha, gamma):

		print('Testing...\n')

		best_W = self.find_W(self.X_trainval_gzsl, self.gt_trainval, self.trainval_sig, alpha, gamma) # combine train and val
		
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
			acc_seen_classes = self.zsl_acc_gzsl_f1(self.X_test_seen, best_W, self.labels_test_seen, np.unique(self.labels_test), self.test_sig)
			acc_unseen_classes = self.zsl_acc_gzsl_f1(self.X_test_unseen, best_W, self.labels_test_unseen, np.unique(self.labels_test), self.test_sig)
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
	
	clf = ESZSL(args)
	
	if args.mode=='train': 
		args.alpha, args.gamma = clf.fit()
	else:
		if args.dataset=='CUB':
			args.alpha, args.gamma = 3, 0
		if args.dataset=='AWA1':
			args.alpha, args.gamma = 3, 0
		if args.dataset=='AWA2':
			args.alpha, args.gamma = 3, 0
		if args.dataset=='APY':
			args.alpha, args.gamma = 2, 0
		if args.dataset=='SUN':
			args.alpha, args.gamma = 3, 2
	
	clf.evaluate(args.alpha, args.gamma)
