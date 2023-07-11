lista = []
def oblicz_pole_powierzchni_3D(lista):
    pole = 0
    liczba_punktow = len(lista)

    for i in range(liczba_punktow):
        p1 = lista[i]
        p2 = lista[(i + 1) % liczba_punktow]
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        pole += (y2 - y1) * (z2 + z1)
        pole += (z2 - z1) * (x2 + x1)
        pole += (x2 - x1) * (y2 + y1)
        pole = abs(pole) / 2
    return pole
    
with open("136587.txt", "r") as plik:
    for linia in plik:
        x, y, z = map(float,linia.strip().split())
        lista.append((x, y, z))
    pole_powierzchni = oblicz_pole_powierzchni_3D(lista)
    print("Pole powierzchni 3D wynosi:",pole_powierzchni)
    plik.close()