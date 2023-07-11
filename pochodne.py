import numpy as np
import matplotlib.pyplot as plt


dane = np.loadtxt('136587.txt')


zy = dane[:,2]
z = np.zeros((11,21))
k=0
for i in range(11):
    for j in range(21):
        z[i,j]=zy[k]
        k=k+1

m = 11
n = 21

def pochodna(n,m,z):
    x = np.linspace(0,2,n)
    y = np.linspace(0,1,m)
    x,y = np.meshgrid(x,y)

    pz = z[0]
    px = x[0]
    zx = np.zeros(n)

    for i in range(n):
        if(i == 0):
            zx[i] = (pz[i+1] - pz[i]) / (px[i+1] - px[i])
        elif(i == n-1):
            zx[i] = (pz[i] - pz[i-1]) / (px[i] - px[i-1])
        else:
            zx[i] = (pz[i+1] - pz[i-1]) / (px[i+1] - px[i-1])

    plt.plot(px, zx, 'green', label="Pochodne cząstkowe")
    plt.xlabel('x')
    plt.ylabel('f(x,y)')
    plt.title('Pochodne cząstkowe')
    plt.legend()
    plt.show()

    return zx

def monotonicznosc(pochodne,x):
    for i in range(len(pochodne)-1):
        if (pochodne[i] > 0 and pochodne[i+1] <0):
            print("Maks w okolicach punktu x =", x[i])
        elif (pochodne[i] < 0 and pochodne[i+1] > 0):
            print("Min w okolicach punktu x =", x[i])
        if (pochodne[i] > 0):
            print("Funkcja rośnie w poblizu x = ", x[i])
        if (pochodne[i] < 0):
            print("Funkcja maleje w poblizu x = ", x[i])

    if (pochodne[-1] < 0):
        print("Funkcja maleje w poblizu x = ", x[-1])
    else:
        print("Funkcja rośnie w poblizu x = ", x[-1])

x = dane[:,0]
pochodne = pochodna(n,m,z)
pochodne.reshape(1,21)

monotonicznosc(pochodne, x)

print(pochodna(n,m,z))