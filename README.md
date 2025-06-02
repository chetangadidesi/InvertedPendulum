# Inverted Pendulum Control: PID vs LQR Comparative Study

A comprehensive implementation and comparison of classical PID and modern LQR control strategies for the inverted pendulum system. This project demonstrates the fundamental differences between error-based feedback control and optimal state-feedback control.

## Features
- Implements state-space modeling of an inverted pendulum system
- Designs and tunes a PID controller
- Designs an LQR controller using the Riccati equation
- Simulates both control strategies step-by-step
- Generates a side-by-side animated comparison of both methods
- Plots pendulum angle, cart position, and control input over time

## How it works
How It Works

System Modeling:
- The inverted pendulum system is modeled using Newtonian mechanics and converted into a state-space representation.
PID Control:
- A classical PID controller stabilizes the pendulum by minimizing the angle error. Anti-windup is included to avoid integral term saturation.
LQR Control:
- An optimal LQR controller is designed by solving the continuous-time Algebraic Riccati Equation.
Simulation:
- The system is simulated using the Euler method (PID) and odeint (LQR). States and control inputs are recorded.
Animation:
- A side-by-side visualization is created using matplotlib.animation, showing cart and pendulum motion over time.
