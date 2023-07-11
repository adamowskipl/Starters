import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("x.txt")
y = np.loadtxt("z.txt") 
    
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
   


    plt.subplots()
    plt.plot(xtf, ytf, 'b-', linewidth=2.0,label='Splajn')                                     
    plt.plot(x, y, 'ro',label = "Punkty wejsciowe")
    plt.xlabel('x')
    plt.ylabel('Bi(x)')
    plt.title("B-splajn")
    plt.legend()
    plt.show()

inter_splajn(x,y)