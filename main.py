import numpy as np
from rocket_dynamics import rk4_step, f
from atmospheric import atmosphere
from utils import export_csv, plot_flight_data
import os

def simulate_rocket(h0=0.0, v0=0.0, m0=1000.0, t0=0.0, tf=60.0, dt=0.01):
    y = np.array([h0, v0, m0])
    t = t0

    times = []
    altitudes = []
    velocities = []
    machs = []
    masses = []

    while t <= tf and y[0] >= 0:
        rho, a = atmosphere(y[0])
        mach = abs(y[1]) / a if a > 0 else 0

        times.append(t)
        altitudes.append(y[0])
        velocities.append(y[1])
        machs.append(mach)
        masses.append(y[2])

        y = rk4_step(t, y, dt, f)
        t += dt

    print(f"Data points collected: {len(times)}")
    return times, altitudes, velocities, machs, masses

def main():
    os.makedirs("data", exist_ok=True)

    times, altitudes, velocities, machs, masses = simulate_rocket()

    data = list(zip(times, altitudes, velocities, machs, masses))
    header = ["Time (s)", "Altitude (m)", "Velocity (m/s)", "Mach Number", "Mass (kg)"]
    filename = "data/orbitalx_flight_data.csv"

    export_csv(filename, data, header)

    print(f"Simulation data exported to {filename}")

    plot_flight_data(times, altitudes, velocities, machs)

if __name__ == "__main__":
    main()
