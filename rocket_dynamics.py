import numpy as np
from atmospheric import atmosphere

# Constants
g0 = 9.80665  # m/s²

# Rocket initial parameters
m0 = 1000.0   # kg (initial mass)
m_dot = 5.0   # kg/s (fuel burn rate)
thrust = 15000.0  # N (constant thrust)
burn_time = (m0 - 500) / m_dot  # burn until 500 kg mass

Cd = 0.5
A = 1.0

def drag_force(rho, v, Cd=Cd, A=A):
    return 0.5 * rho * v**2 * Cd * A

def f(t, y):
    """
    y = [altitude, velocity, mass]
    """
    h, v, m = y
    if h < 0:
        h = 0

    rho, a = atmosphere(h)
    D = drag_force(rho, v)
    g_force = g0

    # Thrust active?
    T = thrust if t <= burn_time and m > 500 else 0.0

    # Mass change
    m_dot_actual = m_dot if t <= burn_time and m > 500 else 0.0

    dhdt = v
    dvdt = (T - D - m * g_force) / m
    dmdt = -m_dot_actual

    return np.array([dhdt, dvdt, dmdt])

def rk4_step(t, y, dt, derivs=f):
    k1 = derivs(t, y)
    k2 = derivs(t + dt / 2, y + dt * k1 / 2)
    k3 = derivs(t + dt / 2, y + dt * k2 / 2)
    k4 = derivs(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
