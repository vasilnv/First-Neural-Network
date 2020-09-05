from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

'''
import matplotlib.pyplot as plt 
plt.gray()
plt.matshow(digits.images[3])
plt.show()
'''
#print(digits.data)

from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X=X_scale.fit_transform(digits.data)
#print(X[0,:])

from sklearn.model_selection import train_test_split
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


import numpy as np 
def convert_y_to_vector(y):
	y_vector = np.zeros((len(y), 10))
	for i in range(len(y)):
		y_vector[i,y[i]] = 1
	return y_vector

y_v_train = convert_y_to_vector(y_train)
y_v_test = convert_y_to_vector(y_test)

print(y_v_train[0])
print(y_v_test[0])

nn_structure = (64, 30, 10)

def f(x):
	return 1 / (1 + np.exp(-x))

def f_deriv(x):
	return f(x) * (1 - f(x))

import numpy.random as r 
def setup_and_init_weights (nn_structure):
	W = {}
	b = {}
	for l in range(1, len(nn_structure)):
		W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
		b[l] = r.random_sample((nn_structure[l],))
	return W,b

def init_tri_values(nn_structure):
	tri_W = {}
	tri_b = {}

	for l in  range(1, len(nn_structure)):
		tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
		tri_b[l] = np.zeros((nn_structure[l],))
	return tri_W, tri_b

def feed_forward(x, W, b):
	h = {1:x}
	z={}
	for l in range(1, len(W) + 1):
		if l == 1:
			node_in = x
		else:
			node_in = h[l]

		z[l + 1] = W[l].dot(node_in) + b[l]
		h[l + 1] = f(z[l + 1])
	return h,z

def calculate_out_layer_delta(y, h_out, z_out):
	return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
	return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

def train_nn(nn_structure, X, y, iter_num = 3000, alpha = 0.25):
	W, b = setup_and_init_weights(nn_structure)
	cnt = 0
	m = len(y)
	avg_cost_func = []
	print('Starting gradient descent for {} iterations'. format(iter_num))
	while cnt < iter_num:
		if cnt % 1000 == 0:
			print('Iteration {} of {}'.format(cnt, iter_num))
		tri_W, tri_b = init_tri_values(nn_structure)
		avg_cost = 0

		for i in range(len(y)):
			delta = {}

			h,z = feed_forward(X[i, :], W, b)
			for l in range (len(nn_structure), 0, -1):
				if l == len(nn_structure):
					delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
					avg_cost += np.linalg.norm((y[i,:]-h[l]))
				else:
					if l > 1:
						delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
					tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
					tri_b[l] += delta [l+1]

		for l in range(len(nn_structure)-1, 0, -1):
			W[l] += -alpha * (1.0/m * tri_W[l])
			b[l] += -alpha * (1.0/m * tri_b[l])

		avg_cost = 1.0/m * avg_cost
		avg_cost_func.append(avg_cost)
		cnt += 1
	return W, b, avg_cost_func

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

import matplotlib.pyplot as plt 
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()	

def predict_y(W,b,X,n_layers):
	m = X.shape[0]
	y = np.zeros((m,))
	for i in range(m):
		h, z = feed_forward(X[i,:], W, b)
		y[i] = np.argmax(h[n_layers])
	return y

from sklearn.metrics import accuracy_score
y_pred = predict_y(W, b, X_test, 3)
print(y_pred)
print(y_v_test)