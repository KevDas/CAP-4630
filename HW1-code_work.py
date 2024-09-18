#1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#2
data = pd.read_csv('./data_nonlinear.csv')
plt.scatter(data.X, data.Y)
X = data.X
Y = data.Y
plt.xlabel("X")
plt.ylabel("Y")
#plt.show()

#3
# Coding here
a, b, c, d = 0, 0, 0, 0 #model initialization
L = 0.000001
epochs = 10000 

n = float(len(X))

for i in range (epochs):
    Y_pred = a*X**3 + b*X**2 + c*X + d
   
    D_a = (-2/n) * sum(X**3 * (Y - Y_pred)) 
    D_b = (-2/n) * sum(X**2 * (Y - Y_pred))
    D_c = (-2/n) * sum(X * (Y - Y_pred))
    D_d = (-2/n) * sum(Y - Y_pred)
    
    
    a -= L * D_a
    b -= L * D_b
    c -= L * D_c
    d -= L * D_d

        



Y_pred = a*(X**3) + b*(X**2) + c*X + d
plt.scatter(X, Y)
# Scatter plot of original data
plt.scatter(X, Y, label='Original Data')

# Plot of predicted values (for each point in X, compute its Y_pred and plot)
plt.scatter(X, Y_pred, color='red', label='Prediction')

plt.show()

data_set2 = pd.read_csv("./data_two_variables.csv")

X1 = data_set2.X1
X2 = data_set2.X2
Y = data_set2.Y

fig = plt.figure()
ax = plt.axes(projection = "3d")

ax.scatter3D(X1, X2, Y)
plt.show()


a, b, c = 0, 0, 0
L = 0.001
epochs = 10000

n = float(len(X))

for i in range (epochs):
    Y_pred = a*X1 + b*X2 + c
    
    D_a = (-2/n) * sum(X1 * (Y - Y_pred))
    D_b = (-2/n) * sum(X2 * (Y - Y_pred))
    D_c = (-2/n) * sum(Y - Y_pred)
    
    a-= L * D_a
    b -= L * D_b
    c -= L * D_c
    
    if (i < 5):
        print(a, b, c)
        
        
#test insert text        
def L_prediction (a, b, c, X1, X2):
    return a* X1 + b * X2 + c

fig = plt.figure()
ax = plt.axes(projection = "3d")
Y_pred = L_prediction(a, b, c, X1, X2)
ax.scatter3D(X1, X2, Y)
ax.scatter3D(X1, X2, Y_pred, color="red", label="Prediction")
plt.show()