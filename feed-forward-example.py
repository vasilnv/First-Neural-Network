import numpy as np 

w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4],[0.6, 0.6, 0.6]])
w2 = np.zeros((1,3))
w2[0,:] = np.array([0.5, 0.5, 0.5])

b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

def f(x):
	return 1 / (1 + np.exp(-x))

w = [w1, w2]
b = [b1, b2]
#a dummy x input vector
x = [1.5, 2.0, 3.0]
'''
def my_foo(n_layers, x, w, b):
	for l in range(n_layers - 1):
		if l == 0:
			input = x
		else:
			input = output

		output = np.zeros(w[l].shape[0])

		for i in range(w[l].shape[0]):
			sum=0
			for j in range(w[l].shape[1]):
				sum+=w[l][i][j] * input[j]
			sum += b[l][i]
			output[i] = f(sum)
	return output 
'''

def simple_looped_nn_calc(n_layers, x, w, b):
	for l in range(n_layers - 1):
		# all layers
		if l == 0:
			input = x
		else:
			input = output

		#setup the output array
		output = np.zeros((w[l].shape[0]))

		for i in range(w[l].shape[0]):
			f_sum = 0

			for j in range(w[l].shape[1]):
				f_sum += w[l][i][j] * input[j]

			f_sum += b[l][i]

			output[i] = f(f_sum)
	return output

# a.dot(b) - multiplication of matrices

#fast forward function using matrices
def my_foo(n_layers, x, w, b):
	for l in range(n_layers - 1):
		if l == 0:
			input = x
		else:
			input = h

		z=w[l].dot(input) + b[l]
		h=f(z)
	return h
print(my_foo(3, x, w, b))
