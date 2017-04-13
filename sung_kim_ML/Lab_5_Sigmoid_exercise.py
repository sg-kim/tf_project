import matplotlib.pyplot as plt
import math

def sig(x):

    sig_x = math.pow((1 + math.exp(-1*x)), -1)
    return sig_x

x_val = []
y_val = []

for x in range(-100, 101):

    y = sig(x*0.1)

    x_val.append(x)
    y_val.append(y)


plt.plot(x_val, y_val)
plt.show()
