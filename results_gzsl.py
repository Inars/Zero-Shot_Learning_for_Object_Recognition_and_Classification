import math
import numpy as np
from scipy import io, linalg
import matplotlib.pyplot as plt
import argparse
import torch
import operator

parser = argparse.ArgumentParser(description="results")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-calc', '--calculate', help='choose between DS, voting, MDT, DNN, GT, consensus, auction, all', default='all', type=str)
parser.add_argument('-c', '--constant', default=10, type=int)
parser.add_argument('-tol', '--tolerance', default=1.0, type=float)

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

		test_seen_loc = 'test_seen_loc'
		test_unseen_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_test_seen = feat[:, np.squeeze(att_splits[test_seen_loc]-1)]
		self.X_test_unseen = feat[:, np.squeeze(att_splits[test_unseen_loc]-1)]

		labels = res101['labels']
		self.labels_test_seen = np.squeeze(labels[np.squeeze(att_splits[test_seen_loc]-1)])
		self.labels_test_unseen = np.squeeze(labels[np.squeeze(att_splits[test_unseen_loc]-1)])
		self.labels_test = np.concatenate((self.labels_test_seen, self.labels_test_unseen), axis=0)
		
		self.test_classes_seen = np.unique(self.labels_test_seen)
		self.test_classes_unseen = np.unique(self.labels_test_unseen)
		test_classes = np.unique(self.labels_test) # All Classes of the dataset

		print('Test Seen:{}; Test Unseen:{}\n'.format(self.X_test_seen.shape[1], self.X_test_unseen.shape[1]))

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.test_sig = sig[:, test_classes-1] # Entire Signature Matrix

	def calculate_voting(self, classes_devise, classes_ale, classes_eszsl, classes_sae, classes_sje, y_true, tolerance):

		print('Calculating votes...\n')

		classes = np.unique(self.labels_test)
		
		classes_devise_modified = []
		classes_ale_modified = []
		classes_eszsl_modified = []
		classes_sae_modified = []
		classes_sje_modified = []
		labels_test = y_true
		labels_test_modified = []

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
				if tolerated / len(labels_test) < tolerance:
					classes_devise_modified.append(classes_devise[i])
					classes_ale_modified.append(classes_ale[i])
					classes_eszsl_modified.append(classes_eszsl[i])
					classes_sae_modified.append(classes_sae[i])
					classes_sje_modified.append(classes_sje[i])
					labels_test_modified.append(labels_test[i])
					tolerated += 1

		predicted_classes = np.zeros_like(labels_test_modified)

		for i in range(len(labels_test_modified)):
			test_labels_votes = {ele: 0 for ele in classes}
			test_labels_votes[classes_devise_modified[i]] += 1
			test_labels_votes[classes_ale_modified[i]] += 1
			test_labels_votes[classes_eszsl_modified[i]] += 1
			test_labels_votes[classes_sae_modified[i]] += 1
			test_labels_votes[classes_sje_modified[i]] += 1
			predicted_classes[i] = max(test_labels_votes.items(), key=operator.itemgetter(1))[0]

		acc = self.zsl_acc(labels_test_modified, predicted_classes, np.unique(labels_test))

		return acc
			
	def calculate_MDT(self, dist_devise, dist_ale, dist_eszsl, dist_sae, dist_sje, y_true, tolerance):
			
		print('Calculating MDT...\n')

		classes = np.unique(self.labels_test)
		
		dist_devise_modified = []
		dist_ale_modified = []
		dist_eszsl_modified = []
		dist_sae_modified = []
		dist_sje_modified = []
		labels_test = y_true
		labels_test_modified = []

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
				if tolerated / len(labels_test) < tolerance:
					dist_devise_modified.append(dist_devise[i])
					dist_ale_modified.append(dist_ale[i])
					dist_eszsl_modified.append(dist_eszsl[i])
					dist_sae_modified.append(dist_sae[i])
					dist_sje_modified.append(dist_sje[i])
					labels_test_modified.append(labels_test[i])
					tolerated += 1

		prob_dist_devise = np.zeros_like(dist_devise_modified)
		prob_dist_ale = np.zeros_like(dist_ale_modified)
		prob_dist_eszsl = np.zeros_like(dist_eszsl_modified)
		prob_dist_sae = np.zeros_like(dist_sae_modified)
		prob_dist_sje = np.zeros_like(dist_sje_modified)
		for i in range(len(dist_devise_modified)):
			prob_dist_devise[i] = self.conf(dist_devise_modified[i])
			prob_dist_ale[i] = self.conf(dist_ale_modified[i])
			prob_dist_eszsl[i] = self.conf(dist_eszsl_modified[i])
			prob_dist_sae[i] = self.conf(dist_sae_modified[i])
			prob_dist_sje[i] = self.conf(dist_sje_modified[i])
		p_models = np.array([prob_dist_devise, prob_dist_ale, prob_dist_eszsl, prob_dist_sae, prob_dist_sje])
		predicted_classes = np.zeros(len(dist_devise_modified), dtype=np.int16)

		avg_sums = np.zeros(5)
		avg_entropies = np.zeros(5)

		for n in range(len(p_models)):
			ss = 0
			for i in range(len(p_models[n])):
				s = 0
				for j in range(len(p_models[n][i])):
					s += p_models[n][i][j]*math.log(p_models[n][i][j] if p_models[n][i][j] > 0 else 1,2)
				ss += (-s)

			avg_entropy = 0 if len(p_models[n]) == 0 else ss / len(p_models[n])
			avg_sum = 0 if len(p_models[n]) == 0 else sum(np.amax(p_models[n], axis=1)) / len(p_models[n])

			avg_sums[n] = avg_sum
			avg_entropies[n] = avg_entropy

		'''
		fix them?
		APY:
			max_displacement = 0.0101
			entropy_displacement = 0.159171
		AWA1:
			max_displacement = 0.013
			entropy_displacement = 0.16632
		AWA2:
			max_displacement = 0.185
			entropy_displacement = 0.185
		CUB:
			max_displacement = 1
			entropy_displacement = 1
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
					entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j] if p_models[indx][i][j] > 0 else 1,2)
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
							entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j] if p_models[indx][i][j] > 0 else 1,2)
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
									entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j] if p_models[indx][i][j] > 0 else 1,2)
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
											entropy += p_models[indx][i][j]*math.log(p_models[indx][i][j] if p_models[indx][i][j] > 0 else 1,2)
										entropy = -entropy
										if entropy <= avg_entropies[indx] - entropy_displacement:
											predicted_classes[i] = np.argmax(p_models[indx][i])
										else:

											indx = np.argpartition(avg_sums, kth=-1, axis=-1)[-4:-3][0]
											predicted_classes[i] = np.argmax(p_models[indx][i])

		acc = self.zsl_acc(labels_test_modified, predicted_classes, classes)

		return acc

	def calculate_DNN(self, dist_devise, dist_ale, dist_eszsl, dist_sae, dist_sje, y_true, tolerance):
			
		print('Calculating DNN...\n')

		classes = np.unique(self.labels_test)

		dist_devise_modified = []
		dist_ale_modified = []
		dist_eszsl_modified = []
		dist_sae_modified = []
		dist_sje_modified = []
		labels_test = y_true
		labels_test_modified = []

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
				if tolerated / len(labels_test) < tolerance:
					dist_devise_modified.append(dist_devise[i])
					dist_ale_modified.append(dist_ale[i])
					dist_eszsl_modified.append(dist_eszsl[i])
					dist_sae_modified.append(dist_sae[i])
					dist_sje_modified.append(dist_sje[i])
					labels_test_modified.append(labels_test[i])
					tolerated += 1
		
		dist_models = np.array([dist_devise_modified, dist_ale_modified, dist_eszsl_modified, dist_sae_modified, dist_sje_modified])
		predicted_classes = np.zeros_like(labels_test_modified)

		X = np.zeros((len(dist_devise_modified), 5), dtype=np.double)
		y = np.zeros((len(dist_devise_modified), 1), dtype=np.double)

		for i in range(len(dist_devise_modified)):
			X[i][0] = np.argmax(dist_devise_modified[i])
			X[i][1] = np.argmax(dist_ale_modified[i])
			X[i][2] = np.argmax(dist_eszsl_modified[i])
			X[i][3] = np.argmax(dist_sae_modified[i])
			X[i][4] = np.argmax(dist_sje_modified[i])
			if np.argmax(dist_devise_modified[i]) == labels_test_modified[i]:
				y[i] = 0
			elif np.argmax(dist_ale_modified[i]) == labels_test_modified[i]:
				y[i] = 1
			elif np.argmax(dist_eszsl_modified[i]) == labels_test_modified[i]:
				y[i] = 2
			elif np.argmax(dist_sae_modified[i]) == labels_test_modified[i]:
				y[i] = 3
			elif np.argmax(dist_sje_modified[i]) == labels_test_modified[i]:
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
			predicted_models[i] = 1 if np.isnan(predicted_models[i]) else predicted_models[i]
			predicted_classes[i] = np.argmax(dist_models[int(predicted_models[i]) if int(predicted_models[i]) <= 4 else 4][i])

		acc = self.zsl_acc(labels_test_modified, predicted_classes, classes)
		
		return acc

	def calculate_GT(self, dist_devise, dist_ale, dist_eszsl, dist_sae, dist_sje, y_true, tolerance):
			
		print('Calculating GT...\n')

		classes = np.unique(self.labels_test)

		dist_devise_modified = []
		dist_ale_modified = []
		dist_eszsl_modified = []
		dist_sae_modified = []
		dist_sje_modified = []
		labels_test = y_true
		labels_test_modified = []

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
				if tolerated / len(labels_test) < tolerance:
					dist_devise_modified.append(dist_devise[i])
					dist_ale_modified.append(dist_ale[i])
					dist_eszsl_modified.append(dist_eszsl[i])
					dist_sae_modified.append(dist_sae[i])
					dist_sje_modified.append(dist_sje[i])
					labels_test_modified.append(labels_test[i])
					tolerated += 1

		prob_dist_devise = np.zeros_like(dist_devise_modified)
		prob_dist_ale = np.zeros_like(dist_ale_modified)
		prob_dist_eszsl = np.zeros_like(dist_eszsl_modified)
		prob_dist_sae = np.zeros_like(dist_sae_modified)
		prob_dist_sje = np.zeros_like(dist_sje_modified)
		for i in range(len(dist_devise_modified)):
			prob_dist_devise[i] = self.conf(dist_devise_modified[i])
			prob_dist_ale[i] = self.conf(dist_ale_modified[i])
			prob_dist_eszsl[i] = self.conf(dist_eszsl_modified[i])
			prob_dist_sae[i] = self.conf(dist_sae_modified[i])
			prob_dist_sje[i] = self.conf(dist_sje_modified[i])

		dist_models = np.array([prob_dist_devise, prob_dist_ale, prob_dist_eszsl, prob_dist_sae, prob_dist_sje])
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

					dist_models[np.isnan(dist_models)] = 0
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
					break

		acc = self.zsl_acc(labels_test_modified, predicted_classes, classes)

		return acc
	
	def calculate_consensus(self, dist_devise, dist_ale, dist_eszsl, dist_sae, dist_sje, y_true, tolerance):
			
		print('Calculating consensus...\n')

		classes = np.unique(self.labels_test)

		dist_devise_modified = []
		dist_ale_modified = []
		dist_eszsl_modified = []
		dist_sae_modified = []
		dist_sje_modified = []
		labels_test = y_true
		labels_test_modified = []

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
				if tolerated / len(labels_test) < tolerance:
					dist_devise_modified.append(dist_devise[i])
					dist_ale_modified.append(dist_ale[i])
					dist_eszsl_modified.append(dist_eszsl[i])
					dist_sae_modified.append(dist_sae[i])
					dist_sje_modified.append(dist_sje[i])
					labels_test_modified.append(labels_test[i])
					tolerated += 1

		prob_dist_devise = np.zeros_like(dist_devise_modified)
		prob_dist_ale = np.zeros_like(dist_ale_modified)
		prob_dist_eszsl = np.zeros_like(dist_eszsl_modified)
		prob_dist_sae = np.zeros_like(dist_sae_modified)
		prob_dist_sje = np.zeros_like(dist_sje_modified)
		for i in range(len(dist_devise_modified)):
			prob_dist_devise[i] = self.conf(dist_devise_modified[i])
			prob_dist_ale[i] = self.conf(dist_ale_modified[i])
			prob_dist_eszsl[i] = self.conf(dist_eszsl_modified[i])
			prob_dist_sae[i] = self.conf(dist_sae_modified[i])
			prob_dist_sje[i] = self.conf(dist_sje_modified[i])

		dist_models = np.array([prob_dist_devise, prob_dist_ale, prob_dist_eszsl, prob_dist_sae, prob_dist_sje])
		predicted_classes = np.zeros(len(dist_models[0]), dtype=np.int16)
		c = len(labels_test_modified)

		if tolerance > 0:
			for i in range(len(dist_models[0])):
				U = np.zeros((5,5), dtype=np.float)
				W = np.zeros((5,5), dtype=np.float)
				for n in range(5):
					for m in range(5):
						if n == m:
							value = max(dist_models[n][i]) if max(dist_models[n][i]) != 0 else 1
							U[n][m] = 1**c * max(dist_models[n][i]) * math.log(np.absolute(value), c)
						else:
							value = dist_models[n][i][np.argmax(dist_models[m][i])] if dist_models[n][i][np.argmax(dist_models[m][i])] != 0 else 1
							U[n][m] = 1**c * dist_models[n][i][np.argmax(dist_models[m][i])] * math.log(np.absolute(value), c)
				for n in range(5):
					for m in range(5):
						W[n][m] = (1 / (U[m][n]**2 * sum([2*(1/(U[k][n]**2)) for k in range(5)]))) if np.around(U[m][n]**2 * sum([2*(1/(U[k][n]**2)) for k in range(5)]), decimals=5) == 0 else 0
				w, v = linalg.eig(W)
				
				predicted_classes[i] = np.argmax(dist_models[np.argmax(w)][i])
			
			acc = self.zsl_acc(labels_test_modified, predicted_classes, classes)
		else:
			acc = 1.0


		return acc
	
	def calculate_auction(self, dist_devise, dist_ale, dist_eszsl, dist_sae, dist_sje, y_true, tolerance, c):
			
		print('Calculating auction...\n')

		classes = np.unique(self.labels_test)

		dist_devise_modified = []
		dist_ale_modified = []
		dist_eszsl_modified = []
		dist_sae_modified = []
		dist_sje_modified = []
		labels_test = y_true
		labels_test_modified = []

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
				if tolerated / len(labels_test) < tolerance:
					dist_devise_modified.append(dist_devise[i])
					dist_ale_modified.append(dist_ale[i])
					dist_eszsl_modified.append(dist_eszsl[i])
					dist_sae_modified.append(dist_sae[i])
					dist_sje_modified.append(dist_sje[i])
					labels_test_modified.append(labels_test[i])
					tolerated += 1

		prob_dist_devise = np.zeros_like(dist_devise_modified)
		prob_dist_ale = np.zeros_like(dist_ale_modified)
		prob_dist_eszsl = np.zeros_like(dist_eszsl_modified)
		prob_dist_sae = np.zeros_like(dist_sae_modified)
		prob_dist_sje = np.zeros_like(dist_sje_modified)
		for i in range(len(dist_devise_modified)):
			prob_dist_devise[i] = self.conf(dist_devise_modified[i])
			prob_dist_ale[i] = self.conf(dist_ale_modified[i])
			prob_dist_eszsl[i] = self.conf(dist_eszsl_modified[i])
			prob_dist_sae[i] = self.conf(dist_sae_modified[i])
			prob_dist_sje[i] = self.conf(dist_sje_modified[i])

		dist_models = np.array([prob_dist_devise, prob_dist_ale, prob_dist_eszsl, prob_dist_sae, prob_dist_sje])
		dist_models[np.isnan(dist_models)] = 0
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

		acc = self.zsl_acc(labels_test_modified, predicted_classes, classes)

		return acc

	def conf(self, vector):
		e = np.exp(vector)
		return (e / e.sum()) if e.sum() != 0 and e.sum() != math.inf else 0

	def conf(self, vector):
		res = np.zeros_like(vector)
		for i in range(len(vector)):
			res[i] = vector[i]/vector.sum()
		return res

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

	def evaluate(self):

		dist_devise_seen = np.loadtxt("testing/gzsl/devise_dist_seen_"+self.args.dataset+".txt")
		dist_ale_seen = np.loadtxt("testing/gzsl/ale_dist_seen_"+self.args.dataset+".txt")
		dist_eszsl_seen = np.loadtxt("testing/gzsl/eszsl_dist_seen_"+self.args.dataset+".txt")
		dist_sae_seen = np.loadtxt("testing/gzsl/sae_dist_seen_"+self.args.dataset+".txt")
		dist_sje_seen = np.loadtxt("testing/gzsl/sje_dist_seen_"+self.args.dataset+".txt")
		
		dist_devise_unseen = np.loadtxt("testing/gzsl/devise_dist_unseen_"+self.args.dataset+".txt")
		dist_ale_unseen = np.loadtxt("testing/gzsl/ale_dist_unseen_"+self.args.dataset+".txt")
		dist_eszsl_unseen = np.loadtxt("testing/gzsl/eszsl_dist_unseen_"+self.args.dataset+".txt")
		dist_sae_unseen = np.loadtxt("testing/gzsl/sae_dist_unseen_"+self.args.dataset+".txt")
		dist_sje_unseen = np.loadtxt("testing/gzsl/sje_dist_unseen_"+self.args.dataset+".txt")

		classes_devise_seen = np.loadtxt("testing/gzsl/devise_pred_seen_"+self.args.dataset+".txt")
		classes_ale_seen = np.loadtxt("testing/gzsl/ale_pred_seen_"+self.args.dataset+".txt")
		classes_eszsl_seen = np.loadtxt("testing/gzsl/eszsl_pred_seen_"+self.args.dataset+".txt")
		classes_sae_seen = np.loadtxt("testing/gzsl/sae_pred_seen_"+self.args.dataset+".txt")
		classes_sje_seen = np.loadtxt("testing/gzsl/sje_pred_seen_"+self.args.dataset+".txt")

		classes_devise_unseen = np.loadtxt("testing/gzsl/devise_pred_unseen_"+self.args.dataset+".txt")
		classes_ale_unseen = np.loadtxt("testing/gzsl/ale_pred_unseen_"+self.args.dataset+".txt")
		classes_eszsl_unseen = np.loadtxt("testing/gzsl/eszsl_pred_unseen_"+self.args.dataset+".txt")
		classes_sae_unseen = np.loadtxt("testing/gzsl/sae_pred_unseen_"+self.args.dataset+".txt")
		classes_sje_unseen = np.loadtxt("testing/gzsl/sje_pred_unseen_"+self.args.dataset+".txt")

		test_classes_seen = self.labels_test_seen
		test_classes_unseen = self.labels_test_unseen

		if args.calculate == "voting":
			acc_seen = clf.calculate_voting(classes_devise_seen, classes_ale_seen, classes_eszsl_seen, classes_sae_seen, classes_sje_seen, test_classes_seen, self.args.tolerance)
			acc_unseen = clf.calculate_voting(classes_devise_unseen, classes_ale_unseen, classes_eszsl_unseen, classes_sae_unseen, classes_sje_unseen, test_classes_unseen, self.args.tolerance)
			acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
			print("Voting acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))
		if args.calculate == "MDT":
			acc_seen = clf.calculate_MDT(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, self.args.tolerance)
			acc_unseen = clf.calculate_MDT(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, self.args.tolerance)
			acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
			print("MDT acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))
		if args.calculate == "DNN":
			acc_seen = clf.calculate_DNN(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, self.args.tolerance)
			acc_unseen = clf.calculate_DNN(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, self.args.tolerance)
			acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
			print("DNN acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))
		if args.calculate == "GT":
			acc_seen = clf.calculate_GT(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, self.args.tolerance)
			acc_unseen = clf.calculate_GT(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, self.args.tolerance)
			acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
			print("GT acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))
		if args.calculate == "consensus":
			acc_seen = clf.calculate_consensus(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, self.args.tolerance)
			acc_unseen = clf.calculate_consensus(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, self.args.tolerance)
			acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
			print("Consensus acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))
		if args.calculate == "auction":
			acc_seen = clf.calculate_auction(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, self.args.tolerance, self.args.constant)
			acc_unseen = clf.calculate_auction(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, self.args.tolerance, self.args.constant)
			acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
			print("Auction acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))
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

				acc_seen = clf.calculate_voting(classes_devise_seen, classes_ale_seen, classes_eszsl_seen, classes_sae_seen, classes_sje_seen, test_classes_seen, tolerance[i])
				acc_unseen = clf.calculate_voting(classes_devise_unseen, classes_ale_unseen, classes_eszsl_unseen, classes_sae_unseen, classes_sje_unseen, test_classes_unseen, tolerance[i])
				acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
				voting_acc.append(acc)
				print("Voting acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))

				acc_seen = clf.calculate_MDT(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, tolerance[i])
				acc_unseen = clf.calculate_MDT(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, tolerance[i])
				acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
				MDT_acc.append(acc)
				print("MDT acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))

				acc_seen = clf.calculate_DNN(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, tolerance[i])
				acc_unseen = clf.calculate_DNN(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, tolerance[i])
				acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
				DNN_acc.append(acc)
				print("DNN acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))

				acc_seen = clf.calculate_GT(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, tolerance[i])
				acc_unseen = clf.calculate_GT(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, tolerance[i])
				acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
				GT_acc.append(acc)
				print("GT acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))

				acc_seen = clf.calculate_consensus(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, tolerance[i])
				acc_unseen = clf.calculate_consensus(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, tolerance[i])
				acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
				consensus_acc.append(acc)
				print("Consensus acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))

				acc_seen = clf.calculate_auction(dist_devise_seen, dist_ale_seen, dist_eszsl_seen, dist_sae_seen, dist_sje_seen, test_classes_seen, tolerance[i], self.args.constant)
				acc_unseen = clf.calculate_auction(dist_devise_unseen, dist_ale_unseen, dist_eszsl_unseen, dist_sae_unseen, dist_sje_unseen, test_classes_unseen, tolerance[i], self.args.constant)
				acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
				auction_acc.append(acc)
				print("Auction acc: {}; Seen: {}; Unseen: {}".format(np.round(acc,decimals=4), acc_seen, acc_unseen))

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
			plt.savefig('results/plot_'+args.dataset+'_gzsl.pdf', bbox_inches='tight')
			plt.savefig('results/plot_'+args.dataset+'_gzsl.png', bbox_inches='tight')
			plt.show()


if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = Results(args)	
	clf.evaluate()