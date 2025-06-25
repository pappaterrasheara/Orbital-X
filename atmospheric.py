import numpy as np

g0 = 9.80665
R = 287.05
gamma = 1.4
T0 = 288.15
P0 = 101325
L = 0.0065
H_trop = 11000

def atmosphere(h):
    if h < H_trop:
        T = T0 - L * h
        P = P0 * (T / T0) ** (g0 / (R * L))
    else:
        T = T0 - L * H_trop
        P = P0 * ((T / T0) ** (g0 / (R * L))) * np.exp(-g0 * (h - H_trop) / (R * T))
    rho = P / (R * T)
    a = np.sqrt(gamma * R * T)
    return rho, a
