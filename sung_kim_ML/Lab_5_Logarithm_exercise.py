import matplotlib.pyplot as plt
import math

def log10(x):
    return math.log(x, 10)

y_val = []
x_val = []

for x in range(1, 101):

    #pow_x = math.pow(10, x)
    #y = log10(pow_x)
    y = log10(x)

    x_val.append(x)
    y_val.append(y)

    #print('x: %.3f y: %.3f'%(pow_x, y))

plt.plot(x_val, y_val)
plt.show()
