import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("x.txt")
y = np.loadtxt("z.txt")

def aproksymacja_f1zm(x, y):
    xt = x
    yt = y
    M = np.zeros((2,2))
    B = np.zeros((2,1))
    A = np.zeros((2,1))

    s1=0
    s2=0
    ysum=0
    xysum=0
    for i in range(xt.shape[0]):
        s1+=xt[i]
        s2+=(xt[i])**2
        ysum+=yt[i]
        xysum+=xt[i]*yt[i]

    M[0][0]=xt.shape[0]
    M[0][1]=s1
    M[1][0]=s1
    M[1][1]=s2
    B[0][0]=ysum
    B[1][0]=xysum
    A=np.linalg.solve(M,B)
    return A

def aproksymacja_f1zmKwadrat(x,y):
    xt=x
    yt=y
    M = np.zeros((3,3))
    B = np.zeros((3,1))
    A = np.zeros((3,1))
    s1=0
    s2=0
    s3=0
    s4=0
    ysum=0
    xysum=0
    x2ysum=0
    for i in range(xt.shape[0]):
        s1+=xt[i]
        s2+=(xt[i])**2
        s3+=(xt[i])**3
        s4+=(xt[i])**4
        ysum+=y[i]
        xysum+=x[i]*y[i]
        x2ysum+=((x[i])**2)*y[i]
    M[0][0]=xt.shape[0]
    M[1][0]=s1
    M[2][0]=s2
    M[0][1]=s1
    M[1][1]=s2
    M[2][1]=s3
    M[0][2]=s2
    M[1][2]=s3
    M[2][2]=s4

    B[0]=ysum
    B[1]=xysum
    B[2]=x2ysum
    A=np.linalg.solve(M,B)
    return A

a=aproksymacja_f1zm(x,y)
a1=aproksymacja_f1zmKwadrat(x,y)
xf=np.linspace(0,2,100)
yf=np.zeros(xf.shape[0])
yf2=np.zeros(xf.shape[0])
for i in range(yf.shape[0]):
    yf[i]=a[0]+a[1]*xf[i]
for i in range(yf2.shape[0]):
    yf2[i]=a1[0]+a1[1]*xf[i]+a1[2]*(xf[i])**2
    
def oblicz_calka_aproksymacja(x, y, A):
    xf = np.linspace(0, 2, 100)
    yf = np.zeros(xf.shape[0])
    for i in range(yf.shape[0]):
        if A.shape[0] == 2:
            yf[i] = A[0] + A[1] * xf[i]
        elif A.shape[0] == 3:
            yf[i] = A[0] + A[1] * xf[i] + A[2] * (xf[i]) ** 2

    calka = np.trapz(yf, xf)
    return calka

plt.subplots()
plt.plot(xf,yf,'r-',linewidth=2.0,label='f(x)')
plt.plot(xf,yf2,'y-',linewidth=2.0,label='g(x)')
plt.plot(x,y,'go')
plt.xlabel('Wspolrzedne x')
plt.ylabel('Wspolrzedne y')
plt.title("Aproksymacja funkcji jednej zmiennej")
plt.legend()
plt.grid()
plt.show()

calka_f = oblicz_calka_aproksymacja(x, y, a)
calka_g = oblicz_calka_aproksymacja(x, y, a1)

print("Całka z funkcji f(x):", calka_f)
print("Całka z funkcji g(x):", calka_g)