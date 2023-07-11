import numpy as np
import matplotlib.pyplot as plt
import math

tab = np.loadtxt("136587.txt")
x = tab[:, 0]
y = tab[:,1]
z = tab[:,2]

lista = []


def mapy(tab):
    X = np.linspace(0.0, 2.0, 21)
    Y = np.linspace(0.0, 1.0, 11)
    X = np.around(X, 1)
    Y = np.around(Y, 1)
    xx, yy = np.meshgrid(X, Y)
    z = tab[:, 2]
    zz = np.reshape(z, (yy.shape[0], xx.shape[1]))

    vmin = np.min(z)
    vmax = np.max(z)
    wykres = plt.pcolormesh(xx, yy, zz, vmin=vmin, vmax=vmax)
    plt.colorbar(wykres, label="wartości z")
    plt.title("Mapa 2D")
    plt.xlabel("Współrzędne x")
    plt.ylabel("Współrzędne y")
    plt.show()
    
    fig  = plt.figure()
    ax = plt.axes(projection = '3d')
    mapa = ax.plot_surface(xx,yy,zz,cmap ="viridis")
    ax.set_title("Mapa 3D")
    ax.set_xlabel("Wspolrzedne x")
    ax.set_ylabel("Wspolrzedne y")
    ax.set_zlabel("Wspolrzedne z")
    ax.set_box_aspect([1.0,0.5,0.5])
    ax.plot_wireframe(xx,yy,zz,color = "green",alpha = 0.3)
    fig.colorbar(mapa,label="wartosci z")
    plt.show()
     
  
def tworzenie_tablicy2(lista):
    for i in range(231):
        lista.append(z[i])

tworzenie_tablicy2(lista)

def srednia(lista):
    o = 0
    for i in range(0, len(lista), 21):
        srednia = sum(lista[i:i+21])/21
        print(f"Srednia dla wartosci y {round(o,1)}: {round(srednia,3)}")
        o+= 0.1
        srednia = 0
        
    print("Srednia z calosci: ",round(sum(lista)/len(lista),3),"\n")

def odchylenie(lista):
    o = 0
    for i in range(0, len(lista), 21):
        lista_temp = lista[i:i+21]
        srednia = sum(lista_temp) / len(lista_temp)
        suma_kwadratow = sum((x - srednia) ** 2 for x in lista_temp)
        wariancja = suma_kwadratow / len(lista_temp)
        odchylenie = math.sqrt(wariancja)
        print(f"Odchylenie standardowe dla wartości y {round(o, 1)}: {round(odchylenie, 8)}")
        o += 0.1
        
    srednia = sum(lista) / len(lista)
    suma_kwadratow = sum((x - srednia) ** 2 for x in lista)
    wariancja = suma_kwadratow / len(lista)
    odchylenie = math.sqrt(wariancja)
    print("Odchylenie z całej listy:", round(odchylenie, 8),"\n")


    

def mediana(lista):
   n = len(lista)
   s = sorted(lista)
   return (s[n//2-1]/2.0+s[n//2]/2.0, s[n//2])[n % 2] if n else None

o = 0
for i in range(0,len(x)-20,21):
    print("Mediana dla wartosci y: ",round(o,1), mediana(z[i:i+21]))
    o+=0.1
        
mediana_calosci = lista[len(lista)//2]
if len(lista) % 2 == 0:
    mediana_calosci = (lista[len(lista)//2 - 1] + lista[len(lista)//2]) / 2
    
print("Mediana z calosci: ", round(mediana_calosci, 3))

    
mapy(tab)
srednia(lista)
odchylenie(lista)
mediana(lista)
