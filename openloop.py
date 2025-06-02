# INVERTED PENDULUM

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
from scipy import signal
    

def define_system_parameters():
    params = {
        'M' : 0.5, # Mass of cart
        'm' : 0.2, #Mass of pendulum
        'b' : 0.1, #Coeff of friction
        'l' : 0.3, #Length of pendulum
        'I' : 0.006, #Moment of Inertia
        'g' : 9.81   # gravity
    }
    return params

def create_transfer_functions(params):
    """Create transfer functions P_cart and P_pend like in MATLAB"""
    
    # Extract parameters (matching your MATLAB values exactly)
    M = params['M']
    m = params['m']
    b = params['b'] 
    I = params['I']
    g = params['g']
    l = params['l']
    
    # Calculate q (denominator) - matches your MATLAB
    q = (M + m) * (I + m*l**2) - (m*l)**2
    
    print(f"q = {q:.6f}")
    
    # P_cart transfer function: from force u to cart position x
    # Numerator: ((I+m*l^2)/q)*s^2 - (m*g*l/q)
    P_cart_num = np.array([((I + m*l**2)/q), 0, -(m*g*l/q)])
    
    # P_pend transfer function: from force u to pendulum angle phi  
    # Numerator: (m*l*s/q)
    P_pend_num = np.array([(m*l/q), 0])
    
    # Common denominator for both: s^4 + (b*(I + m*l^2))*s^3/q - ((M + m)*m*g*l)*s^2/q - b*m*g*l*s/q
    common_den = np.array([
        1,                                    # s^4 coefficient
        (b*(I + m*l**2))/q,                  # s^3 coefficient  
        -((M + m)*m*g*l)/q,                  # s^2 coefficient
        -(b*m*g*l)/q,                        # s^1 coefficient
        0                                     # s^0 coefficient
    ])
    
    # Adjust P_pend denominator (it's s^3, not s^4)
    P_pend_den = np.array([
        1,                                    # s^3 coefficient
        (b*(I + m*l**2))/q,                  # s^2 coefficient
        -((M + m)*m*g*l)/q,                  # s^1 coefficient  
        -(b*m*g*l)/q                         # s^0 coefficient
    ])
    
    # Create transfer function objects
    P_cart = signal.TransferFunction(P_cart_num, common_den)
    P_pend = signal.TransferFunction(P_pend_num, P_pend_den)
    
    return P_cart, P_pend

def step_response_transfer_function(params):
    """Generate step response using transfer functions like MATLAB lsim"""
    
    # Create transfer functions
    P_cart, P_pend = create_transfer_functions(params)
    
    # Time vector (matching your MATLAB: t = 0:0.05:10)
    t = np.arange(0, 10.05, 0.05)  # 0 to 10 seconds, step 0.05
    
    # Step input (matching your MATLAB: u = ones(size(t)))
    u = np.ones_like(t)
    
    # Simulate step response for both transfer functions
    # This is equivalent to MATLAB's lsim(sys_tf, u, t)
    t_cart, y_cart, _ = signal.lsim(P_cart, u, t)
    t_pend, y_pend, _ = signal.lsim(P_pend, u, t)
    
    return t, y_cart, y_pend

def create_state_space_matrices(params):
    M = params['M']
    m = params['m']
    b = params['b']
    l = params['l']
    I = params['I']
    g = params['g']
    
    p = I * (M + m) + M * m * l**2
    
    # Create A matrix (4x4)
    A = np.array([
        [0, 1, 0, 0],
        [0, -(I + m*l**2) * b / p, (m**2 * g * l**2) / p, 0],
        [0, 0, 0, 1],
        [0, -(m * l * b) / p, m * g * l * (M + m) / p, 0]
    ])
    
    # Create B matrix (4x1)
    B = np.array([
        [0],
        [(I + m*l**2) / p],
        [0],
        [m * l / p]
    ])
    
    # Create C matrix (2x4)
    C = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    
    # Create D matrix (2x1)
    D = np.array([
        [0],
        [0]
    ])
    
    return A, B, C, D

def print_state_matrices(A, B, C, D, params):
    print("System Parameters: ")
    print(f"Cart mass (M): {params['M']} kg")
    print(f"Pendulum mass (m): {params['m']} kg")
    print(f"Coefficient of Friction (b): {params['b']} N*s/m")
    print(f"Length of Pendulum (l): {params['l']} m")
    print(f"Moment of Inertia(I): {params['I']} kg*m^2")
    print(f"Acceleration due to gravity(g): {params['g']} m/s^2")
    print()
    
    print("A matrix (4x4):")
    print(A)
    print()
    
    print("B matrix (4x1):")
    print(B.flatten())  
    print()
    
    print("C matrix (2x4):")
    print(C)
    print()
    
    print("D matrix (2x1):")
    print(D.flatten())
    print()
    
def state_space_dynamics(x, t, u, A, B):
    """State Space Dynamics function: dx/dt = Ax + Bu

    Args:
        x : state vector [cart_position, cart_velocity, pendulum_angle, pendulum_velocity]
        t : time 
        u : control input
        A : state space matrix
        B : state space matrix
    
    Returns:
        dx_dt : state derivative
    """
    x = np.array(x).reshape(-1,1) #Making sure x is a column vector
    u = np.array([[u]]) #Making sure u is scalar in matrix form
    dx_dt = A @ x + B @ u
    return dx_dt.flatten() #Return as 1D array


def simulate_open_loop(A, B, C, D, initial_state, time_span, control_input=1):
    """Simulate the system with open loop

    Args:
        A : state matrix
        B : state matrix
        C : state matrix
        D : state matrix
        initial_state : [x0, x_dot0, theta0, theta_dot0]
        time_span : array of time points
        control_input : constant force applied to cart
        
    Returns:
        t : time array
        states : state history
        outputs : output history
    """
    # Simulate using odeint
    states = odeint(state_space_dynamics, initial_state, time_span, 
                   args=(control_input, A, B))
    
    # Calculate outputs y = Cx + Du
    outputs = []
    for state in states:
        state_vec = state.reshape(-1, 1)
        u_vec = np.array([[control_input]])
        y = C @ state_vec + D @ u_vec
        outputs.append(y.flatten())
    
    outputs = np.array(outputs)
    
    return time_span, states, outputs

def animate_pendulum(time, states, params):
    """Create animation of the inverted pendulum"""
    cart_pos = states[:, 0]  # Cart position
    pendulum_angle = states[:, 2]  # Pendulum angle
    l = params['l']  # Pendulum length
    
    # Calculate pendulum bob position
    bob_x = cart_pos + l * np.sin(pendulum_angle)
    bob_y = l * np.cos(pendulum_angle)
    
    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Animation subplot
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Inverted Pendulum Animation (No Control - Should Fall!)')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Height (m)')
    
    # Initialize animation elements
    cart, = ax1.plot([], [], 'bs', markersize=20, label='Cart')
    pendulum, = ax1.plot([], [], 'r-', linewidth=3, label='Pendulum')
    bob, = ax1.plot([], [], 'ro', markersize=8, label='Bob')
    ground, = ax1.plot([-2, 2], [-0.1, -0.1], 'k-', linewidth=2)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.legend(loc='upper right')
    
    # Response plots
    ax2.plot(time, cart_pos, 'b-', label='Cart Position (m)', linewidth=2)
    ax2.plot(time, pendulum_angle, 'r-', label='Pendulum Angle (rad)', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Response')
    ax2.set_title('System Response')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Animation function
    def animate(frame):
        if frame < len(time):
            # Current positions
            x_cart = cart_pos[frame]
            x_bob = bob_x[frame]
            y_bob = bob_y[frame]
            
            # Update cart position
            cart.set_data([x_cart], [0])
            
            # Update pendulum rod
            pendulum.set_data([x_cart, x_bob], [0, y_bob])
            
            # Update pendulum bob
            bob.set_data([x_bob], [y_bob])
            
            # Update time display
            time_text.set_text(f'Time: {time[frame]:.2f} s\nAngle: {pendulum_angle[frame]:.2f} rad')
            
        return cart, pendulum, bob, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time), 
                                 interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    return fig, anim

    


if __name__ == "__main__":
    #Initialize system
    params = define_system_parameters()
    A, B, C, D = create_state_space_matrices(params)
    
    #Print matrices to verify
    print_state_matrices(A, B, C, D, params)
    
    # Test simulation
    # Initial state: [cart_pos, cart_vel, pendulum_angle, pendulum_vel]
    initial_state = [0, 0, 0, 0]  # Small initial angle (0.1 rad ≈ 5.7 degrees)
    
    # Time span
    t = np.linspace(0, 5, 500)  # 5 seconds, 500 points
    
    #Control input
    control_input = 1
    
    # Simulate the system
    time, states, outputs = simulate_open_loop(A, B, C, D, initial_state, t, control_input)
    
    # Extract outputs: outputs[:, 0] is cart position, outputs[:, 1] is pendulum angle
    y_cart = outputs[:, 0]  # Cart position
    y_pend = outputs[:, 1]  # Pendulum angle
    
    
    # Create animations
    print("Creating animation for scenario 1 (small perturbation)...")
    fig1, anim1 = animate_pendulum(time, states, params)
    # Plot results 
    plt.figure(figsize=(12, 8))
    plt.plot(time, y_cart, 'b-', label='x (cart position)', linewidth=2)
    plt.plot(time, y_pend, 'r-', label='phi (pendulum angle)', linewidth=2)
    plt.title('Open-Loop Step Response (State-Space Method)')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.xlim([0, 3])  # Match your MATLAB axis([0 3 0 50])
    plt.ylim([0, 50])
    plt.legend()
    plt.grid(True)
    plt.show()
    
    



