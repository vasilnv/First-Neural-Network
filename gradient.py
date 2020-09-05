x_old = 0
x_new = 6
gamma = 0.01
precision = 0.000001

def df(x):
	y = 4 * x**3 - 9 * x**2
	return y

while abs(x_new - x_old) > precision:
	x_old = x_new
	x_new = x_old - gamma * df(x_old)
	print("x_new is:")
	print(x_new)
	print("gradient is:")
	print(df(x_old))

print("The local minimum is at: %f"  % x_new)


