import numpy as np 
import sympy as sp

# Define variables
w = sp.Symbol('w')
x = 1  # given value of x
y = 3 * x - w  # y = 3x - w
z = 5 * y**3  # z = 5y^3
l = (z - 3)**2  # l = (z - 3)^2

# Compute the gradient (derivative of l with respect to w)
grad_l_w = sp.diff(l, w)

# Substitute w = 2 to get the numerical result
grad_l_w_value = grad_l_w.subs(w, 2)

#print(grad_l_w_value)  # Final result



A = np.array([[1,2], [3,4]])
B = np.array([[0,1], [2,3]])
C = np.array([4, 5])


D = (A * B) + C

#print(D)


E = (np.dot(A, B)) + C

#print(E)


AB = np.dot(A, B)

#print(AB)
ABC = np.dot(AB, C)

#print(ABC)


# Define x as a symbol (now we're optimizing with respect to x)
x = sp.Symbol('x')

# Recompute the expressions for y, z, and l with respect to x
y = 3 * x - 2  # y = 3x - w (w = 2 is fixed)
z = 5 * y**3  # z = 5y^3
l = (z - 3)**2  # l = (z - 3)^2

# Compute the gradient (derivative of l with respect to x)
grad_l_x = sp.diff(l, x)

# Given x = 1 (initial value of x)
initial_x = 1
learning_rate = 0.01

# Calculate the gradient at x = 1
grad_l_x_value = grad_l_x.subs(x, initial_x)

# Apply the gradient descent update rule: x_new = x - eta * grad_l_x
new_x = initial_x - learning_rate * grad_l_x_value

print(new_x)  
