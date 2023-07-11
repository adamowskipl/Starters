import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
def f(a,b,c,x):
    return a*x*x+b*x+c

x = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')
z = np.loadtxt('z.txt')
def interp_lag(x,y):
    a0 = y[0]/((x[0]-x[1])*(x[0] - x[2]))
    a1 = y[1]/((x[1]-x[0])*(x[1] - x[2]))
    a2 = y[2]/((x[2]-x[0])*(x[2] - x[1]))
    a = a0 + a1 + a2
    b = a0 * (-x[1]-x[2]) + a1 * (-x[0] - x[2]) + a2 * (-x[0] - x[1])
    c = a0 * (-x[1]*-x[2]) + a1 * (-x[0] *-x[2]) + a2 * (-x[0] * -x[1])
    return [a,b,c]

for i in range(0,20,2):
    a,b,c = interp_lag(x[i:i+3], z[i:i+3])
    xp = np.linspace(x[i], x[i+2], 100)
    plt.plot(xp, f(a,b,c,xp), 'g-',linewidth = 5.0)
    
def B(x,hp):
    h=hp
    B = 0
    if(-2*h<=x<=-h):
         B = (x+2*h)**3
    elif(-1*h <= x <=0):
         B = (h**3) + 3*(h**2) * (x+h)+3*h*((x+h)**2) -3*((x+h)**3)
    elif(0 <= x <=h):
         B = (h**3) + 3*(h**2)*(h-x)+3*h*((h-x)**2) -3*((h-x)**3)
    elif(h <= x <=2*h):
         B = (2*h -x)**3
    else:
         B=0
    return B/(h**3)  

def wyznacz_wartosc(x,xT,k,h):
        suma = 0
        for j in range(xT.shape[0]):
            suma+=k[j]*B(x-xT[j],h)
        return suma

def inter_splajn(x,y):
    np.around(6)
    n=x.shape[0]
    h=x[1]-x[0]
    M=np.zeros((n+2,n+2))
    yx=np.zeros(n+2)
    k=np.zeros(n+2)
    xT=np.zeros(n+2)
    yf=np.zeros(n)
    yx[0]=1
    yx[n+1]=-1
    for i in range (1, n+1):
        yx[i]=y[i-1]
    M[0][0]=-3/h
    M[n+1][n-1]=-3/h
    M[0][2]=3/h
    M[n+1][n+1]=3/h
    for i in range (1,n+1):
        M[i][i]=4
        M[i][i-1]=1
        M[i][i+1]=1
    k=np.linalg.solve(M,yx)
    xT[0]=x[0]-h
    for i in range(1,n+1):
        xT[i]=x[i-1]
    xT[n+1]=x[n-1]+h
    for i in range(x.shape[0]):
        suma = 0
        for j in range(xT.shape[0]):
            suma+=k[j]*B(x[i]-xT[j],h)
        yf[i]=suma
    
    xtf=np.linspace(0,2,100)
    ytf=np.zeros(xtf.shape[0])
    for i in range(xtf.shape[0]):
         ytf[i]=wyznacz_wartosc(xtf[i],xT,k,h)
   


    plt.plot(xtf, ytf, 'r-', linewidth=2.0,label='Splajn')                                     
    plt.plot(x, y, 'ro',label = "Punkty wejsciowe")
    plt.xlabel('x')
    plt.ylabel('Bi(x)')
    return x,y

inter_splajn(x,z)

def oblicz_calka(x, y):
    x3, y3 = inter_splajn(x, y)
    calka = np.trapz(x3, y3)
    return calka

calka = oblicz_calka(x, z)
    
plt.plot(x[0:21],z[0:21],'bo', label="Punkty wejsciowe")
plt.xlabel("Wspolrzedne x")
plt.ylabel("Wspolrzedne y")
plt.title("Interpolacja Lagrange'a - projekt")
plt.legend()
plt.show()

print("Całka z funkcji interpolacyjnych: ", calka)