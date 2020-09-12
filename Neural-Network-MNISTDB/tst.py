import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()

'''
import matplotlib.pyplot as plt 
plt.gray()
plt.matshow(digits.images[3])
plt.show()
'''
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X=X_scale.fit_transform(digits.data)
#print(X[0,:])

from sklearn.model_selection import train_test_split
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

y_vector = np.zeros((5, 10))

import numpy as np 
def convert_y_to_vector(y):
	y_vector = np.zeros((len(y), 10))
	for i in range(len(y)):
		y_vector[i,y[i]] = 1
	return y_vector
y_v_train = convert_y_to_vector(y_train)
y_v_test = convert_y_to_vector(y_test)


nn_structure = (64, 30, 10)

def f(x):
	return 1 / (1 + np.exp(-x))

def f_deriv(x):
	return f(x) * (1 - f(x))

import numpy.random as r 
arr=r.random_sample((2,3))

def setup_and_init_weights (nn_structure):
	W = {}
	b = {}
	for l in range(1, len(nn_structure)):
		W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
		b[l] = r.random_sample((nn_structure[l],))
	return W,b

W,b = setup_and_init_weights(nn_structure)

def init_tri_values(nn_structure):
	tri_W = {}
	tri_b = {}

	for l in  range(1, len(nn_structure)):
		tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
		tri_b[l] = np.zeros((nn_structure[l],))
	return tri_W, tri_b

j={1:5}
p={}
print(p)
print(len(p))
print(type(p))


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

def predict_y(W,b,X,n_layers):
	m = X.shape[0]
	y = np.zeros((m,))
	for i in range(m):
		h, z = feed_forward(X[i,:], W, b)
		y[i] = np.argmax(h[n_layers])
	return y

