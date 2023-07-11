import numpy as np
import matplotlib.pyplot as plt
import math

def Lagrange(x, xi, yi):
    n = len(xi)
    L = [1]*n
    p = 0
    for i in range(n):
        for j in range(n):
            if i!=j:
                L[i] *= (x-xi[j])/(xi[i]-xi[j])
        p += yi[i]*L[i]
    return p

def interp_lag(x, y):
    n = len(x)
    a = [0]*n
    
    for i in range(n):
        p = 1
        for j in range(n):
            if i!=j:
                p *= (x[i]-x[j])
        a[i] = y[i]/p
    
    a_3 = [x[0]*x[0], x[0], 1, x[1]*x[1], x[1], 1, x[2]*x[2], x[2], 1]
    y_3 = [y[0], y[1], y[2]]
    naprawa = np.linalg.solve(np.array(a_3).reshape(3,3), np.array(y_3))
    
    xx = np.linspace(min(x), max(x), 100)
    yy = [Lagrange(i, x, y) for i in xx]
    plt.plot(xx, yy, label='Interpolacja Lagrangea')
    plt.plot(x, y, 'ro', label='Punkty interpolacyjne')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    return a, naprawa
x = [0.826, -2.215, -6.868, -9.429, -13.799]
y = [3.358, -0.401, -0.581, -4.925, -3.51]

a, naprawa = interp_lag(x, y)

print("Zadanie 3:")
print("a = %.6f" % naprawa[0])
print("b = %.6f" % naprawa[1])
print("c = %.6f" % naprawa[2])
print("\n")
print("Zadanie 4:")
for i in range(len(a)):
    print("a%d = %.6f" % (i, a[i]))
