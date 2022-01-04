import math
from re import T
import numpy as np
from scipy import io, linalg
import matplotlib.pyplot as plt
import argparse
import torch

parser = argparse.ArgumentParser(description="results")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-calc', '--calculate', help='choose between DS, voting, MDT, DNN, GT, consensus, auction, all', default='all', type=str)
parser.add_argument('-c', '--constant', default=10, type=int)
parser.add_argument('-tol', '--tolerance', default=1.0, type=float)
parser.add_argument('-seg', '--segregate', default=0, type=int)
parser.add_argument('-lvl', '--level', default=0, type=int)

class TwoLayerNet(torch.nn.Module):
	def __init__(self, D_in, H1, H2, D_out):
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H1)
		self.linear2 = torch.nn.Linear(H1, H2)
		self.linear3 = torch.nn.Linear(H2, D_out)

	def forward(self, X):
		h1_relu = self.linear1(X).clamp(min=0)
		h2_relu = self.linear2(h1_relu).clamp(min=0)
		y_pred = self.linear3(h2_relu)
		return y_pred

class Results():
	
	def __init__(self, args):

		self.args = args

		data_folder = '../xlsa17/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		test_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Test:{}\n'.format(self.X_test.shape[1]))

		labels = res101['labels']
		self.labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])

		test_labels_unseen = np.unique(self.labels_test)
		
		k=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = k
			k+=1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.test_sig = sig[:, test_labels_unseen-1]

		if args.dataset == 'SUN':
			self.attributes = io.loadmat('testing/attributes/attributes_SUN.mat')['attributes']
		else:
			self.attributes = np.genfromtxt('testing/attributes/attributes_'+args.dataset+'.txt', dtype=str)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def softmax(self, vector):
		e = np.exp(vector)
		return e / e.sum()

	def conf(self, vector):
		res = np.zeros_like(vector)
		for i in range(len(vector)):
			res[i] = vector[i]/vector.sum()
		return res

	def calculate(self):
		
		print('Calculating...\n')

		correct_pred_devise = np.loadtxt("testing/zsl/devise_"+args.dataset+".txt")
		correct_pred_ale = np.loadtxt("testing/zsl/ale_"+args.dataset+".txt")
		correct_pred_eszsl = np.loadtxt("testing/zsl/eszsl_"+args.dataset+".txt")
		correct_pred_sae = np.loadtxt("testing/zsl/sae_"+args.dataset+".txt")
		correct_pred_sje = np.loadtxt("testing/zsl/sje_"+args.dataset+".txt")

		print('DEVISE:{} {}; Hard:{} {}\n'.format(len([i for i in correct_pred_devise if i == 1]), 
												len([i for i in correct_pred_devise if i == 1])/len(correct_pred_devise)*100, 
												len([i for i in correct_pred_devise if i == 0]), 
												len([i for i in correct_pred_devise if i == 0])/len(correct_pred_devise)*100))

		print('ALE:{} {}; Hard:{} {}\n'.format(len([i for i in correct_pred_ale if i == 1]), 
												len([i for i in correct_pred_ale if i == 1])/len(correct_pred_ale)*100, 
												len([i for i in correct_pred_ale if i == 0]), 
												len([i for i in correct_pred_ale if i == 0])/len(correct_pred_ale)*100))

		print('SJE:{} {}; Hard:{} {}\n'.format(len([i for i in correct_pred_sje if i == 1]), 
												len([i for i in correct_pred_sje if i == 1])/len(correct_pred_sje)*100, 
												len([i for i in correct_pred_sje if i == 0]), 
												len([i for i in correct_pred_sje if i == 0])/len(correct_pred_sje)*100))

		print('ESZSL:{} {}; Hard:{} {}\n'.format(len([i for i in correct_pred_eszsl if i == 1]), 
												len([i for i in correct_pred_eszsl if i == 1])/len(correct_pred_eszsl)*100, 
												len([i for i in correct_pred_eszsl if i == 0]), 
												len([i for i in correct_pred_eszsl if i == 0])/len(correct_pred_eszsl)*100))

		print('SAE:{} {}; Hard:{} {}\n'.format(len([i for i in correct_pred_sae if i == 1]), 
												len([i for i in correct_pred_sae if i == 1])/len(correct_pred_sae)*100, 
												len([i for i in correct_pred_sae if i == 0]), 
												len([i for i in correct_pred_sae if i == 0])/len(correct_pred_sae)*100))

		very_easy = 0
		very_hard = 0
		other = 0

		easy_att_DEVISE = np.zeros_like(self.test_sig.T[0])
		hard_att_DEVISE = np.zeros_like(self.test_sig.T[0])
		easy_att_ALE = np.zeros_like(self.test_sig.T[0])
		hard_att_ALE = np.zeros_like(self.test_sig.T[0])
		easy_att_SJE = np.zeros_like(self.test_sig.T[0])
		hard_att_SJE = np.zeros_like(self.test_sig.T[0])
		easy_att_ESZSL = np.zeros_like(self.test_sig.T[0])
		hard_att_ESZSL = np.zeros_like(self.test_sig.T[0])
		easy_att_SAE = np.zeros_like(self.test_sig.T[0])
		hard_att_SAE = np.zeros_like(self.test_sig.T[0])

		very_easy_att = np.zeros_like(self.test_sig.T[0])
		very_hard_att = np.zeros_like(self.test_sig.T[0])
		for i in range(len(correct_pred_devise)):
			correct = 0
			incorrect = 0
			class_number = self.labels_test[i]

			if correct_pred_devise[i] == 0:
				incorrect += 1
				hard_att_DEVISE = [sum(x) for x in zip(hard_att_DEVISE, self.test_sig.T[class_number-1])]
			else:
				correct += 1
				easy_att_DEVISE = [sum(x) for x in zip(easy_att_DEVISE, self.test_sig.T[class_number-1])]

			if correct_pred_ale[i] == 0:
				incorrect += 1
				hard_att_ALE = [sum(x) for x in zip(hard_att_ALE, self.test_sig.T[class_number-1])]
			else:
				correct += 1
				easy_att_ALE = [sum(x) for x in zip(easy_att_ALE, self.test_sig.T[class_number-1])]

			if correct_pred_eszsl[i] == 0:
				incorrect += 1
				hard_att_ESZSL = [sum(x) for x in zip(hard_att_ESZSL, self.test_sig.T[class_number-1])]
			else:
				correct += 1
				easy_att_ESZSL = [sum(x) for x in zip(easy_att_ESZSL, self.test_sig.T[class_number-1])]

			if correct_pred_sae[i] == 0:
				incorrect += 1
				hard_att_SAE = [sum(x) for x in zip(hard_att_SAE, self.test_sig.T[class_number-1])]
			else:
				correct += 1
				easy_att_SAE = [sum(x) for x in zip(easy_att_SAE, self.test_sig.T[class_number-1])]

			if correct_pred_sje[i] == 0:
				incorrect += 1
				hard_att_SJE = [sum(x) for x in zip(hard_att_SJE, self.test_sig.T[class_number-1])]
			else:
				correct += 1
				easy_att_SJE = [sum(x) for x in zip(easy_att_SJE, self.test_sig.T[class_number-1])]
			
			if correct == 5:
				very_easy += 1
				very_easy_att = [sum(x) for x in zip(very_easy_att, self.test_sig.T[class_number-1])]
			elif incorrect == 5:
				very_hard += 1
				very_hard_att = [sum(x) for x in zip(very_hard_att, self.test_sig.T[class_number-1])]
			else:
				other += 1
		
		print('Total: Easy:{} {}; Hard:{} {}; Other:{} {}\n'.format(very_easy, very_easy/len(correct_pred_devise)*100, very_hard, very_hard/len(correct_pred_devise)*100, other, other/len(correct_pred_devise)*100))

		if args.dataset == 'SUN':
			print('DEVISE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_DEVISE.index(max(easy_att_DEVISE))][0][0],self.attributes[hard_att_DEVISE.index(max(hard_att_DEVISE))][0][0]))
			print('ALE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_ALE.index(max(easy_att_ALE))][0][0],self.attributes[hard_att_ALE.index(max(hard_att_ALE))][0][0]))
			print('SJE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_SJE.index(max(easy_att_SJE))][0][0],self.attributes[hard_att_SJE.index(max(hard_att_SJE))][0][0]))
			print('ESZSL: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_ESZSL.index(max(easy_att_ESZSL))][0][0],self.attributes[hard_att_ESZSL.index(max(hard_att_ESZSL))][0][0]))
			print('SAE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_SAE.index(max(easy_att_SAE))][0][0],self.attributes[hard_att_SAE.index(max(hard_att_SAE))][0][0]))
			
			print('Total: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[very_easy_att.index(max(very_easy_att))][0][0],self.attributes[very_hard_att.index(max(very_hard_att))][0][0]))
		else:
			print('DEVISE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_DEVISE.index(max(easy_att_DEVISE))][1],self.attributes[hard_att_DEVISE.index(max(hard_att_DEVISE))][1]))
			print('ALE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_ALE.index(max(easy_att_ALE))][1],self.attributes[hard_att_ALE.index(max(hard_att_ALE))][1]))
			print('SJE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_SJE.index(max(easy_att_SJE))][1],self.attributes[hard_att_SJE.index(max(hard_att_SJE))][1]))
			print('ESZSL: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_ESZSL.index(max(easy_att_ESZSL))][1],self.attributes[hard_att_ESZSL.index(max(hard_att_ESZSL))][1]))
			print('SAE: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[easy_att_SAE.index(max(easy_att_SAE))][1],self.attributes[hard_att_SAE.index(max(hard_att_SAE))][1]))
			
			print('Total: Easy Attr:{}; Hard Attr:{}\n'.format(self.attributes[very_easy_att.index(max(very_easy_att))][1],self.attributes[very_hard_att.index(max(very_hard_att))][1]))

	def calculate_voting(self, tolerance, segregate, level):

		print('Calculating votes...\n')

		segregate = False if segregate == 0 else True

		classes_devise_modified, classes_ale_modified, classes_eszsl_modified, classes_sae_modified, classes_sje_modified, labels_test_modified, labels_test = self.load_results(tolerance, shuffle=False, conf=False, probability=False, classes=True, segregate=segregate, level=level)

		test_labels_unseen = np.unique(labels_test)
		predicted_classes = np.zeros_like(labels_test_modified)

		for i in range(len(labels_test_modified)):
			test_labels_votes = np.zeros_like(test_labels_unseen)
			test_labels_votes[list(test_labels_unseen).index(classes_devise_modified[i])] += 1
			test_labels_votes[list(test_labels_unseen).index(classes_ale_modified[i])] += 1
			test_labels_votes[list(test_labels_unseen).index(classes_eszsl_modified[i])] += 1
			test_labels_votes[list(test_labels_unseen).index(classes_sae_modified[i])] += 1
			test_labels_votes[list(test_labels_unseen).index(classes_sje_modified[i])] += 1
			predicted_classes[i] = np.argmax(test_labels_votes)

		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))

		return acc
			
	def calculate_MDT(self, tolerance, segregate, level):
			
		print('Calculating MDT...\n')

		segregate = False if segregate == 0 else True

		p_models, labels_test_modified, labels_test = self.load_results(tolerance, shuffle=False, conf=False, probability=True, classes=False, segregate=segregate, level=level)
		predicted_classes = np.zeros(len(p_models[0]), dtype=np.int16)


		avg_sums = np.zeros(5)
		avg_entropies = np.zeros(5)

		for n in range(len(p_models)):
			ss = 0
			for i in range(len(p_models[n])):
				s = 0
				for j in range(len(p_models[n][i])):
					s += p_models[n][i][j]*math.log(p_models[n][i][j],2)
				ss += (-s)

			avg_entropy = ss / len(p_models[n])
			avg_sum = sum(np.amax(p_models[n], axis=1)) / len(p_models[n])

			avg_sums[n] = avg_sum
			avg_entropies[n] = avg_entropy

		'''
		CUB:
			max_displacement = 1
			entropy_displacement = 1
		AWA1:
			max_displacement = 0.013
			entropy_displacement = 0.16632
		AWA2:
			max_displacement = 0.185
			entropy_displacement = 0.185
		APY:
			max_displacement = 0.0101
			entropy_displacement = 0.159171
		SUN:
			max_displacement = 3.21
			entropy_displacement = 3.21
		'''
		max_displacement = 3.21
		entropy_displacement = 3.21

		for i in range(len(p_models[0])):

			indx = indx = np.argpartition(avg_sums, kth=-1, axis=-1)[-1:][0]
			if max(p_models[indx][i]) > (max(avg_sums) + max_displacement):
				predicted_classes[i] = np.argmax(p_models[indx][i])
			else:
				entropy = 0
				for j in range(len(p_models[indx][i])):
					entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j],2)
				entropy = -entropy
				if entropy <= avg_entropies[indx] - entropy_displacement:
					predicted_classes[i] = np.argmax(p_models[indx][i])
				else:

					indx = np.argpartition(avg_sums, kth=-1, axis=-1)[-2:-1][0]
					if max(p_models[indx][i]) > (max(avg_sums) + max_displacement):
						predicted_classes[i] = np.argmax(p_models[indx][i])
					else:
						entropy = 0
						for j in range(len(p_models[indx][i])):
							entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j],2)
						entropy = -entropy
						if entropy <= avg_entropies[indx] - entropy_displacement:
							predicted_classes[i] = np.argmax(p_models[indx][i])
						else:

							indx = np.argpartition(avg_sums, kth=-1, axis=-1)[-3:-2][0]
							if max(p_models[indx][i]) > (max(avg_sums) + max_displacement):
								predicted_classes[i] = np.argmax(p_models[indx][i])
							else:
								entropy = 0
								for j in range(len(p_models[indx][i])):
									entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j],2)
								entropy = -entropy
								if entropy <= avg_entropies[indx] - entropy_displacement:
									predicted_classes[i] = np.argmax(p_models[indx][i])
								else:

									indx = np.argpartition(avg_sums, kth=-1, axis=-1)[-3:-2][0]
									if max(p_models[indx][i]) > (max(avg_sums) + max_displacement):
										predicted_classes[i] = np.argmax(p_models[indx][i])
									else:
										entropy = 0
										for j in range(len(p_models[indx][i])):
											entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j],2)
										entropy = -entropy
										if entropy <= avg_entropies[indx] - entropy_displacement:
											predicted_classes[i] = np.argmax(p_models[indx][i])
										else:

											indx = np.argpartition(avg_sums, kth=-1, axis=-1)[-4:-3][0]
											predicted_classes[i] = np.argmax(p_models[indx][i])

		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))

		return acc

	def calculate_DNN(self, tolerance, segregate, level):
			
		print('Calculating DNN...\n')

		segregate = False if segregate == 0 else True
		
		dist_models, labels_test_modified, labels_test = self.load_results(tolerance, shuffle=False, conf=False, probability=False, classes=False, segregate=segregate, level=level)
		predicted_classes = np.zeros_like(labels_test_modified)

		X = np.zeros((len(dist_models[0]), 5), dtype=np.double)
		y = np.zeros((len(dist_models[0]), 1), dtype=np.double)

		for i in range(len(dist_models[0])):
			X[i][0] = np.argmax(dist_models[0][i])
			X[i][1] = np.argmax(dist_models[1][i])
			X[i][2] = np.argmax(dist_models[2][i])
			X[i][3] = np.argmax(dist_models[3][i])
			X[i][4] = np.argmax(dist_models[4][i])
			if np.argmax(dist_models[0][i]) == labels_test_modified[i]:
				y[i] = 0
			elif np.argmax(dist_models[1][i]) == labels_test_modified[i]:
				y[i] = 1
			elif np.argmax(dist_models[2][i]) == labels_test_modified[i]:
				y[i] = 2
			elif np.argmax(dist_models[3][i]) == labels_test_modified[i]:
				y[i] = 3
			elif np.argmax(dist_models[4][i]) == labels_test_modified[i]:
				y[i] = 4
			else:
				y[i] = 1

		X = torch.from_numpy(X)
		y = torch.from_numpy(y)
		
		D_in, H1, H2, D_out = len(X[0]), 500, 50, 1

		model = TwoLayerNet(D_in, H1, H2, D_out)

		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
		for t in range(1000):
			y_pred = model(X.float())
			
			loss = criterion(y_pred, y.float())
			#print(t, loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		dist = model(X.float())

		predicted_models = np.around(np.amax(dist.detach().numpy(), axis=1))

		for i in range(len(predicted_models)):
			predicted_classes[i] = np.argmax(dist_models[int(predicted_models[i]) if int(predicted_models[i]) < 5 else 4][i])

		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))

		return acc

	def calculate_GT(self, tolerance, segregate, level):
			
		print('Calculating GT...\n')

		segregate = False if segregate == 0 else True

		dist_models, labels_test_modified, labels_test = self.load_results(tolerance, shuffle=False, conf=True, probability=True, classes=False, segregate=segregate, level=level)
		predicted_classes = np.zeros(len(dist_models[0]), dtype=np.int16)

		for i in range(len(dist_models[0])):
			L = np.array([0, 1, 2, 3, 4], dtype=np.int16) # negotiation list
			M = np.zeros((2,2), dtype=np.float) # payoff matrix
			max_conf = np.array([max(dist_models[0][i]),max(dist_models[1][i]),max(dist_models[2][i]),max(dist_models[3][i]),max(dist_models[4][i])]) # vector of largest confidence value for each base classifier
			L = L[max_conf.argsort()[::-1]] # sorts L in a descending order based on the values of max_conf
			alpha = 0.9

			j = 0
			while j < len(L):
				if max_conf[L[j]] < alpha:
					L = np.delete(L, j)
				else:
					j += 1
			L_prime = L
			n, m = 0, 0 # n: first base model index, m: second base model index
			counter = 0 # counter to break the while loop if it gets stuck

			while len(L) > 1:
				if counter == 2:
					counter = 0
					break
				if np.array_equal(L, L_prime):
					if m >= len(L) - 1:
						if n >= len(L) -2:
							n, m = 0, 1 # n: first base model index, m: second base model index
							counter += 1
						else:
							n += 1
							m = n + 1
					else:
						m += 1
				else:
					n, m = 0, 1 # n: first base model index, m: second base model index
				L_prime = L

				M[0][0] = (dist_models[L[n]][i][np.argmax(dist_models[L[n]][i])] + dist_models[L[n]][i][np.argmax(dist_models[L[m]][i])]) / 2
				M[0][1] = dist_models[L[n]][i][np.argmax(dist_models[L[n]][i])] - dist_models[L[n]][i][np.argmax(dist_models[L[m]][i])]
				M[1][0] = (dist_models[L[m]][i][np.argmax(dist_models[L[m]][i])] + dist_models[L[m]][i][np.argmax(dist_models[L[n]][i])]) / 2
				M[1][1] = dist_models[L[m]][i][np.argmax(dist_models[L[m]][i])] - dist_models[L[m]][i][np.argmax(dist_models[L[n]][i])]

				conf_1 = 0
				conf_2 = 0

				if M[0][0] >= M[0][1]:
					dist_models[L[n]][i][np.argmax(dist_models[L[n]][i])] = M[0][0]
					conf_1 = M[0][0]
				else:
					dist_models[L[n]][i][np.argmax(dist_models[L[n]][i])] = M[0][1]
					conf_1 = M[0][1]
				if M[1][0] >= M[1][1]:
					dist_models[L[m]][i][np.argmax(dist_models[L[m]][i])] = M[1][0]
					conf_2 = M[1][0]
				else:
					dist_models[L[m]][i][np.argmax(dist_models[L[m]][i])] = M[1][1]
					conf_2 = M[1][1]

				if not ((conf_1 == M[0][0] and conf_2 == M[1][0]) or (conf_1 == M[0][1] and conf_2 == M[1][1])):
					if conf_1 == M[0][0]:
						L = np.delete(L, n)
					else:
						L = np.delete(L, m)
				
				M = np.zeros((2,2), dtype=np.float) # payoff matrix
				j = 0
				while j < len(L):
					if max_conf[L[j]] < alpha:
						L = np.delete(L, j)
					else:
						j += 1

				max_conf = np.array([])
				for j in range(len(L)):
					max_conf = np.append(max_conf, max(dist_models[L[j]][i]))
				L = L[max_conf.argsort()[::-1]] # sorts L in a descending order based on the values of max_conf
				max_conf = np.array([max(dist_models[0][i]),max(dist_models[1][i]),max(dist_models[2][i]),max(dist_models[3][i]),max(dist_models[4][i])]) # vector of largest confidence value for each base classifier
		
			if len(L) == 0:
				L = np.array([0, 1, 2, 3, 4], dtype=np.int16) # negotiation list
				L = L[max_conf.argsort()[::-1]] # sorts L in a descending order based on the values of max_conf
			
			predicted_classes[i] = np.argmax(dist_models[L[0]][i])

		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))

		'''segregate = False if segregate == 0 else True

		dist_models, labels_test_modified, labels_test = self.load_results(tolerance, shuffle=False, conf=True, probability=True, classes=False, segregate=segregate, level=level)
		predicted_classes = np.zeros(len(dist_models[0]), dtype=np.int16)

		for i in range(len(dist_models[0])):
			ranking = np.zeros((5), dtype=np.int16)
			dist = np.array([max(dist_models[0][i]),max(dist_models[1][i]),max(dist_models[2][i]),max(dist_models[3][i]),max(dist_models[4][i])])
			indices  = np.array([list(dist_models[0][i]).index(max(dist_models[0][i])),
								list(dist_models[1][i]).index(max(dist_models[1][i])),
								list(dist_models[2][i]).index(max(dist_models[2][i])),
								list(dist_models[3][i]).index(max(dist_models[3][i])),
								list(dist_models[4][i]).index(max(dist_models[4][i]))])

			for j in range(5):
				ranking[list(dist).index(max(dist))] = j + 1
				dist[list(dist).index(max(dist))] = -1000
			
			while(ranking[0]>0 or ranking[1]>0 or ranking[2]>0 or ranking[3]>0 or ranking[4]>0):
				if sum(ranking) > -15:
					[n, m] = ranking.argsort()[-2:][::-1]
					keep_payoff_ag1 = dist_models[n][i][indices[n]] - dist_models[n][i][indices[m]]
					keep_payoff_ag2 = dist_models[m][i][indices[m]] - dist_models[n][i][indices[n]]
					change_payoff_ag1 = (dist_models[n][i][indices[n]] + dist_models[n][i][indices[m]])/2
					change_payoff_ag2 = (dist_models[m][i][indices[m]] + dist_models[n][i][indices[n]])/2

					if keep_payoff_ag1 > change_payoff_ag1:
						dist_models[n][i][indices[n]] = keep_payoff_ag1
					else:
						dist_models[n][i][indices[n]] = change_payoff_ag1
						ranking[n] = -5
					if keep_payoff_ag2 > change_payoff_ag2:
						dist_models[m][i][indices[m]] = keep_payoff_ag2
					else:
						dist_models[m][i][indices[m]] = change_payoff_ag2
						ranking[m] = -5

					dist = np.array([max(dist_models[0][i]),max(dist_models[1][i]),max(dist_models[2][i]),max(dist_models[3][i]),max(dist_models[4][i])])
					indices  = np.array([list(dist_models[0][i]).index(max(dist_models[0][i])),
										list(dist_models[1][i]).index(max(dist_models[1][i])),
										list(dist_models[2][i]).index(max(dist_models[2][i])),
										list(dist_models[3][i]).index(max(dist_models[3][i])),
										list(dist_models[4][i]).index(max(dist_models[4][i]))])

				else:
					if ranking[0] > 0:
						predicted_classes[i] = np.argmax(dist_models[0][i])
					if ranking[1] > 0:
						predicted_classes[i] = np.argmax(dist_models[1][i])
					if ranking[2] > 0:
						predicted_classes[i] = np.argmax(dist_models[2][i])
					if ranking[3] > 0:
						predicted_classes[i] = np.argmax(dist_models[3][i])
					if ranking[4] > 0:
						predicted_classes[i] = np.argmax(dist_models[4][i])
					break'''
			
		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))
		
		return acc
	
	def calculate_consensus(self, tolerance, segregate, level):
			
		print('Calculating consensus...\n')

		segregate = False if segregate == 0 else True

		dist_models, labels_test_modified, labels_test = self.load_results(tolerance, shuffle=False, conf=True, probability=True, classes=False, segregate=segregate, level=level)
		predicted_classes = np.zeros(len(dist_models[0]), dtype=np.int16)
		c = len(labels_test_modified)

		for i in range(len(dist_models[0])):
			U = np.zeros((5,5), dtype=np.float)
			W = np.zeros((5,5), dtype=np.float)
			for n in range(5):
				for m in range(5):
					if n == m:
						U[n][m] = 1**c * max(dist_models[n][i]) * math.log(np.absolute(max(dist_models[n][i])), c)
					else:
						U[n][m] = 1**c * dist_models[n][i][np.argmax(dist_models[m][i])] * math.log(np.absolute(dist_models[n][i][np.argmax(dist_models[m][i])]), c)
			for n in range(5):
				for m in range(5):
					W[n][m] = 1 / (U[m][n]**2 * sum([2*(1/(U[k][n]**2)) for k in range(5)]))
			w, v = linalg.eig(W)
			
			predicted_classes[i] = np.argmax(dist_models[np.argmax(w)][i])

		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))

		return acc
	
	def calculate_auction(self, tolerance, segregate, level, c):
			
		print('Calculating auction...\n')

		segregate = False if segregate == 0 else True

		dist_models, labels_test_modified, labels_test = self.load_results(tolerance, shuffle=False, conf=True, probability=True, classes=False, segregate=segregate, level=level)
		predicted_classes = np.zeros(len(dist_models[0]), dtype=np.int16)

		for i in range(len(dist_models[0])):
			auctioneers = np.ones(5, dtype=np.int16)
			for j in range(4):
				cost = np.zeros(5, dtype=np.int16)
				for n in range(5):
					if auctioneers[n] == 0: continue
					for m in range(5):
						if n == m: continue
						if auctioneers[m] == 0: continue
						cost[n] += np.absolute(max(dist_models[n][i]) - dist_models[n][i][np.argmax(dist_models[m][i])])
					cost[n] /= c
				for n in range(5):
					if auctioneers[n] == 0: continue
					dist_models[n][i][np.argmax(dist_models[n][i])] -= cost[n]
				auctioneers[np.argmax(cost)] = 0
			predicted_classes[i] = np.argmax(dist_models[np.argmax(auctioneers)][i])

		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))

		return acc

	def zsl_acc(self, y_true, y_pred, classes): # Class Averaged Top-1 Accuarcy
		acc = 0
		for i in range(len(classes)):
			correct_predictions = 0
			samples = 0
			for j in range(len(y_true)):
				if y_true[j] == classes[i]:
					samples += 1
					if y_pred[j] == y_true[j]:
						correct_predictions += 1
			if samples == 0:
				acc += 1
			else:
				acc += correct_predictions/samples

		acc = acc/len(classes)

		return acc

	def load_results(self, tolerance, shuffle=False, conf=True, probability=True, classes=False, segregate=False, level=0):
		if classes:
			classes_devise = np.loadtxt("testing/zsl/devise_pred_"+args.dataset+".txt")
			classes_ale = np.loadtxt("testing/zsl/ale_pred_"+args.dataset+".txt")
			classes_eszsl = np.loadtxt("testing/zsl/eszsl_pred_"+args.dataset+".txt")
			classes_sae = np.loadtxt("testing/zsl/sae_pred_"+args.dataset+".txt")
			classes_sje = np.loadtxt("testing/zsl/sje_pred_"+args.dataset+".txt")
			classes_devise_modified = []
			classes_ale_modified = []
			classes_eszsl_modified = []
			classes_sae_modified = []
			classes_sje_modified = []
			labels_test = self.labels_test
			labels_test_modified = []

			if shuffle:
				shuffler = np.random.permutation(len(classes_devise))
				classes_devise = classes_devise[shuffler]
				classes_ale = classes_ale[shuffler]
				classes_eszsl = classes_eszsl[shuffler]
				classes_sae = classes_sae[shuffler]
				classes_sje = classes_sje[shuffler]
				labels_test = labels_test[shuffler]

			tolerated = 0
			for i in range(len(labels_test)):
				if classes_devise[i] == labels_test[i] and \
					classes_ale[i] == labels_test[i] and \
					classes_eszsl[i] == labels_test[i] and \
					classes_sae[i] == labels_test[i] and \
					classes_sje[i] == labels_test[i]:
					classes_devise_modified.append(classes_devise[i])
					classes_ale_modified.append(classes_ale[i])
					classes_eszsl_modified.append(classes_eszsl[i])
					classes_sae_modified.append(classes_sae[i])
					classes_sje_modified.append(classes_sje[i])
					labels_test_modified.append(labels_test[i])
				else:
					if segregate:
						res = [classes_devise[i], classes_ale[i], classes_eszsl[i], classes_sae[i], classes_sje[i]]
						ele, count = np.unique(res, return_counts=True)
						index = np.where(ele == labels_test[i])
						if level > 0 and np.shape(index)[1] > 0:
							if ele[index[0]] == labels_test[i]:
								if count[index[0]] == level:
									if tolerated / len(labels_test) < tolerance:
										classes_devise_modified.append(classes_devise[i])
										classes_ale_modified.append(classes_ale[i])
										classes_eszsl_modified.append(classes_eszsl[i])
										classes_sae_modified.append(classes_sae[i])
										classes_sje_modified.append(classes_sje[i])
										labels_test_modified.append(labels_test[i])
										tolerated += 1
						elif level == 0 and np.shape(index)[1] == 0:
							if tolerated / len(labels_test) < tolerance:
								classes_devise_modified.append(classes_devise[i])
								classes_ale_modified.append(classes_ale[i])
								classes_eszsl_modified.append(classes_eszsl[i])
								classes_sae_modified.append(classes_sae[i])
								classes_sje_modified.append(classes_sje[i])
								labels_test_modified.append(labels_test[i])
								tolerated += 1
					else:
						if tolerated / len(labels_test) < tolerance:
							classes_devise_modified.append(classes_devise[i])
							classes_ale_modified.append(classes_ale[i])
							classes_eszsl_modified.append(classes_eszsl[i])
							classes_sae_modified.append(classes_sae[i])
							classes_sje_modified.append(classes_sje[i])
							labels_test_modified.append(labels_test[i])
							tolerated += 1
			return classes_devise_modified, classes_ale_modified, classes_eszsl_modified, classes_sae_modified, classes_sje_modified, labels_test_modified, labels_test
		else:
			dist_devise = np.loadtxt("testing/zsl/devise_dist_"+args.dataset+".txt")
			dist_ale = np.loadtxt("testing/zsl/ale_dist_"+args.dataset+".txt")
			dist_eszsl = np.loadtxt("testing/zsl/eszsl_dist_"+args.dataset+".txt")
			dist_sae = np.loadtxt("testing/zsl/sae_dist_"+args.dataset+".txt")
			dist_sje = np.loadtxt("testing/zsl/sje_dist_"+args.dataset+".txt")
			dist_devise_modified = []
			dist_ale_modified = []
			dist_eszsl_modified = []
			dist_sae_modified = []
			dist_sje_modified = []
			labels_test = self.labels_test
			labels_test_modified = []

			if shuffle:
				shuffler = np.random.permutation(len(dist_devise))
				dist_devise = dist_devise[shuffler]
				dist_ale = dist_ale[shuffler]
				dist_eszsl = dist_eszsl[shuffler]
				dist_sae = dist_sae[shuffler]
				dist_sje = dist_sje[shuffler]
				labels_test = labels_test[shuffler]

			tolerated = 0
			for i in range(len(labels_test)):
				if np.argmax(dist_devise[i]) == labels_test[i] and \
					np.argmax(dist_ale[i]) == labels_test[i] and \
					np.argmax(dist_eszsl[i]) == labels_test[i] and \
					np.argmax(dist_sae[i]) == labels_test[i] and \
					np.argmax(dist_sje[i]) == labels_test[i]:
					dist_devise_modified.append(dist_devise[i])
					dist_ale_modified.append(dist_ale[i])
					dist_eszsl_modified.append(dist_eszsl[i])
					dist_sae_modified.append(dist_sae[i])
					dist_sje_modified.append(dist_sje[i])
					labels_test_modified.append(labels_test[i])
				else:
					if segregate:
						res = [np.argmax(dist_devise[i]), np.argmax(dist_ale[i]), np.argmax(dist_eszsl[i]), np.argmax(dist_sae[i]), np.argmax(dist_sje[i])]
						ele, count = np.unique(res, return_counts=True)
						index = np.where(ele == labels_test[i])
						if level > 0 and np.shape(index)[1] > 0:
							if ele[index[0]] == labels_test[i]:
								if count[index[0]] == level:
									if tolerated / len(labels_test) < tolerance:
										dist_devise_modified.append(dist_devise[i])
										dist_ale_modified.append(dist_ale[i])
										dist_eszsl_modified.append(dist_eszsl[i])
										dist_sae_modified.append(dist_sae[i])
										dist_sje_modified.append(dist_sje[i])
										labels_test_modified.append(labels_test[i])
										tolerated += 1
						elif level == 0 and np.shape(index)[1] == 0:
							if tolerated / len(labels_test) < tolerance:
								dist_devise_modified.append(dist_devise[i])
								dist_ale_modified.append(dist_ale[i])
								dist_eszsl_modified.append(dist_eszsl[i])
								dist_sae_modified.append(dist_sae[i])
								dist_sje_modified.append(dist_sje[i])
								labels_test_modified.append(labels_test[i])
								tolerated += 1
					else:
						if tolerated / len(labels_test) < tolerance:
							dist_devise_modified.append(dist_devise[i])
							dist_ale_modified.append(dist_ale[i])
							dist_eszsl_modified.append(dist_eszsl[i])
							dist_sae_modified.append(dist_sae[i])
							dist_sje_modified.append(dist_sje[i])
							labels_test_modified.append(labels_test[i])
							tolerated += 1

			dist_models = []
			if probability:
				prob_dist_devise = np.zeros_like(dist_devise_modified)
				prob_dist_ale = np.zeros_like(dist_ale_modified)
				prob_dist_eszsl = np.zeros_like(dist_eszsl_modified)
				prob_dist_sae = np.zeros_like(dist_sae_modified)
				prob_dist_sje = np.zeros_like(dist_sje_modified)
				for i in range(len(dist_devise_modified)):
					if conf:
						prob_dist_devise[i] = self.conf(dist_devise_modified[i])
						prob_dist_ale[i] = self.conf(dist_ale_modified[i])
						prob_dist_eszsl[i] = self.conf(dist_eszsl_modified[i])
						prob_dist_sae[i] = self.conf(dist_sae_modified[i])
						prob_dist_sje[i] = self.conf(dist_sje_modified[i])
					else:
						prob_dist_devise[i] = self.sigmoid(dist_devise_modified[i])
						prob_dist_ale[i] = self.sigmoid(dist_ale_modified[i])
						prob_dist_eszsl[i] = self.sigmoid(dist_eszsl_modified[i])
						prob_dist_sae[i] = self.sigmoid(dist_sae_modified[i])
						prob_dist_sje[i] = self.sigmoid(dist_sje_modified[i])
				dist_models = np.array([prob_dist_devise, prob_dist_ale, prob_dist_eszsl, prob_dist_sae, prob_dist_sje])
			else:
				dist_models = np.array([dist_devise_modified, dist_ale_modified, dist_eszsl_modified, dist_sae_modified, dist_sje_modified])

			
			return dist_models, labels_test_modified, labels_test

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = Results(args)	
	if args.calculate == "DS":
		clf.calculate()
	if args.calculate == "voting":
		acc = np.round(clf.calculate_voting(args.tolerance, args.segregate, args.level),decimals=4)
		print("Voting acc: {}".format(acc))
	if args.calculate == "MDT":
		acc = np.round(clf.calculate_MDT(args.tolerance, args.segregate, args.level),decimals=4)
		print("MDT acc: {}".format(acc))
	if args.calculate == "DNN":
		acc = np.round(clf.calculate_DNN(args.tolerance, args.segregate, args.level),decimals=4)
		print("DNN acc: {}".format(acc))
	if args.calculate == "GT":
		acc = np.round(clf.calculate_GT(args.tolerance, args.segregate, args.level),decimals=4)
		print("GT acc: {}".format(acc))
	if args.calculate == "consensus":
		acc = np.round(clf.calculate_consensus(args.tolerance, args.segregate, args.level),decimals=4)
		print("Consensus acc: {}".format(acc))
	if args.calculate == "auction":
		acc = np.round(clf.calculate_auction(args.tolerance, args.segregate, args.level, args.constant),decimals=4)
		print("Auction acc: {}".format(acc))
	if args.calculate == "all":
		voting_acc = []
		MDT_acc = []
		DNN_acc = []
		GT_acc = []
		consensus_acc = []
		auction_acc = []
		tolerance = np.round(np.arange(0.0, 1.1, 0.1),decimals=2)
		for i in range(len(tolerance)):
			print("Tolerance: {}\n==============\n".format(tolerance[i]))
			voting_acc.append(np.round(clf.calculate_voting(tolerance[i], args.segregate, args.level),decimals=4))
			MDT_acc.append(np.round(clf.calculate_MDT(tolerance[i], args.segregate, args.level),decimals=4))
			DNN_acc.append(np.round(clf.calculate_DNN(tolerance[i], args.segregate, args.level),decimals=4))
			GT_acc.append(np.round(clf.calculate_GT(tolerance[i], args.segregate, args.level),decimals=4))
			consensus_acc.append(np.round(clf.calculate_consensus(tolerance[i], args.segregate, args.level),decimals=4))
			auction_acc.append(np.round(clf.calculate_auction(tolerance[i], args.segregate, args.level, args.constant),decimals=4))
		print("Tolerance: {} {} {} {} {} {} {} {} {} {} {}\nVoting:    {} {} {} {} {} {} {} {} {} {} {}\nMDT:       {} {} {} {} {} {} {} {} {} {} {}\nDNN:       {} {} {} {} {} {} {} {} {} {} {}\nGT:        {} {} {} {} {} {} {} {} {} {} {}\nConsensus: {} {} {} {} {} {} {} {} {} {} {}\nAuction:   {} {} {} {} {} {} {} {} {} {} {}\n"
			   .format(tolerance[0],tolerance[1],tolerance[2],tolerance[3],tolerance[4],tolerance[5],tolerance[6],tolerance[7],tolerance[8],tolerance[9],tolerance[10],
			   			voting_acc[0],voting_acc[1],voting_acc[2],voting_acc[3],voting_acc[4],voting_acc[5],voting_acc[6],voting_acc[7],voting_acc[8],voting_acc[9],voting_acc[10],
						MDT_acc[0],MDT_acc[1],MDT_acc[2],MDT_acc[3],MDT_acc[4],MDT_acc[5],MDT_acc[6],MDT_acc[7],MDT_acc[8],MDT_acc[9],MDT_acc[10],
						DNN_acc[0],DNN_acc[1],DNN_acc[2],DNN_acc[3],DNN_acc[4],DNN_acc[5],DNN_acc[6],DNN_acc[7],DNN_acc[8],DNN_acc[9],DNN_acc[10],
						GT_acc[0],GT_acc[1],GT_acc[2],GT_acc[3],GT_acc[4],GT_acc[5],GT_acc[6],GT_acc[7],GT_acc[8],GT_acc[9],GT_acc[10],
						consensus_acc[0],consensus_acc[1],consensus_acc[2],consensus_acc[3],consensus_acc[4],consensus_acc[5],consensus_acc[6],consensus_acc[7],consensus_acc[8],consensus_acc[9],consensus_acc[10],
						auction_acc[0],auction_acc[1],auction_acc[2],auction_acc[3],auction_acc[4],auction_acc[5],auction_acc[6],auction_acc[7],auction_acc[8],auction_acc[9],auction_acc[10]))

		plt.plot(tolerance, voting_acc, label="Voting")
		plt.plot(tolerance, MDT_acc, label="MDT")
		plt.plot(tolerance, DNN_acc, label="DNN")
		plt.plot(tolerance, GT_acc, label="GT")
		plt.plot(tolerance, consensus_acc, label="Consensus")
		plt.plot(tolerance, auction_acc, label="Auction")
		plt.xlabel('Tolerance')
		plt.ylabel('Accuracy')
		plt.legend()
		name = 'plot_'+args.dataset
		if args.segregate == 1:
			name += '_segragate_' + str(args.level)
		plt.savefig(name+'.pdf', bbox_inches='tight')
		plt.savefig(name+'.png', bbox_inches='tight')
		plt.show()