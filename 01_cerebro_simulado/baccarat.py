import random

def valor_carta(c):
    """En baccarat, J/Q/K/10 valen 0, el as vale 1, el resto su número."""
    return 0 if c >= 10 else c

def mano(cartas):
    """La puntuación es la suma módulo 10."""
    return sum(valor_carta(c) for c in cartas) % 10

def jugar_mano():
    """Juega una mano de baccarat simplificado y devuelve el ganador."""
    mazo = [i for i in range(1, 14)] * 4 * 8  # 8 barajas
    random.shuffle(mazo)
    
    player = [mazo.pop(), mazo.pop()]
    banker = [mazo.pop(), mazo.pop()]
    
    # Regla simplificada de tercera carta
    if mano(player) <= 5:
        player.append(mazo.pop())
    if mano(banker) <= 5:
        banker.append(mazo.pop())
    
    p, b = mano(player), mano(banker)
    if p > b:
        return "player"
    if b > p:
        return "banker"
    return "tie"


# Prueba rápida: juega 10 manos y muestra los resultados
if __name__ == "__main__":
    for i in range(10):
        print(f"Mano {i+1}: ganó {jugar_mano()}")