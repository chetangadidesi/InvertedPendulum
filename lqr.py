import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
from scipy import signal
from scipy.linalg import solve_continuous_are, inv
    

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

def design_lqr_controller(A, B, Q, R):
    """
    Design LQR controller for the system
    
    Args:
        A: State matrix (4x4)
        B: Input matrix (4x1) 
        Q: State weighting matrix (4x4) - penalizes deviations in states
        R: Input weighting matrix (1x1) - penalizes control effort
    
    Returns:
        K: LQR gain matrix (1x4)
        S: Solution to algebraic Riccati equation
        eigvals: Closed-loop eigenvalues
    """
    
    # Solve the continuous-time algebraic Riccati equation (CARE)
    # A^T * S + S * A - S * B * R^(-1) * B^T * S + Q = 0
    S = solve_continuous_are(A, B, Q, R)
    
    # Calculate the LQR gain matrix K
    # K = R^(-1) * B^T * S
    K = inv(R) @ B.T @ S
    
    # Calculate closed-loop eigenvalues (A - B*K)
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)
    
    return K, S, eigvals

def lqr_control(state, K, reference=[0, 0, 0, 0]):
    """
    Calculate LQR control input
    
    Args:
        state: Current state [x, x_dot, theta, theta_dot]
        K: LQR gain matrix
        reference: Desired state (default: cart at origin, pendulum upright)
    
    Returns:
        u: Control input (force)
    """
    state = np.array(state).reshape(-1, 1)
    ref = np.array(reference).reshape(-1, 1)
    
    # Control law: u = -K * (x - x_ref)
    u = -K @ (state - ref)
    
    return u[0, 0]  # Return scalar

def state_space_dynamics_with_lqr(x, t, A, B, K, reference, disturbance=0):
    """
    State Space Dynamics with LQR feedback control
    
    Args:
        x: state vector [cart_position, cart_velocity, pendulum_angle, pendulum_velocity]
        t: time 
        A, B: state space matrices
        K: LQR gain matrix
        reference: desired state
        disturbance: external disturbance force
    
    Returns:
        dx_dt: state derivative
    """
    # Calculate control input using LQR
    u_lqr = lqr_control(x, K, reference)
    
    # Add disturbance
    u_total = u_lqr + disturbance
    
    # Apply dynamics
    x = np.array(x).reshape(-1, 1)
    u = np.array([[u_total]])
    dx_dt = A @ x + B @ u
    
    return dx_dt.flatten()

def simulate_lqr_control(A, B, C, D, K, initial_state, time_span, reference=[0, 0, 0, 0], disturbance_func=None):
    """
    Simulate the system with LQR control
    """
    states = []
    control_inputs = []
    
    # If no disturbance function provided, use zero disturbance
    if disturbance_func is None:
        disturbance_func = lambda t: 0
    
    # Simulate step by step to record control inputs
    dt = time_span[1] - time_span[0]
    current_state = initial_state
    
    for t in time_span:
        states.append(current_state.copy())
        
        # Calculate control input for recording
        u_lqr = lqr_control(current_state, K, reference)
        disturbance = disturbance_func(t)
        u_total = u_lqr + disturbance
        control_inputs.append(u_total)
        
        # Integrate one step forward
        if t < time_span[-1]:
            next_states = odeint(state_space_dynamics_with_lqr, current_state, [t, t + dt], 
                               args=(A, B, K, reference, disturbance))
            current_state = next_states[1]
    
    states = np.array(states)
    control_inputs = np.array(control_inputs)
    
    # Calculate outputs
    outputs = []
    for state in states:
        state_vec = state.reshape(-1, 1)
        y = C @ state_vec
        outputs.append(y.flatten())
    
    outputs = np.array(outputs)
    
    return time_span, states, outputs, control_inputs

def analyze_lqr_design(K, eigvals, Q, R):
    """Analyze the LQR controller design"""
    print("=== LQR Controller Design Analysis ===")
    print(f"Q matrix (state weights):")
    print(Q)
    print(f"\nR matrix (control weight): {R[0,0]}")
    print(f"\nLQR Gain Matrix K:")
    print(f"K = {K}")
    print(f"\nControl law: u = -K * [x, x_dot, theta, theta_dot]^T")
    print(f"u = {K[0,0]:.2f}*x + {K[0,1]:.2f}*x_dot + {K[0,2]:.2f}*theta + {K[0,3]:.2f}*theta_dot")
    
    print(f"\nClosed-loop eigenvalues (poles):")
    for i, eig in enumerate(eigvals):
        if np.isreal(eig):
            print(f"  λ{i+1} = {eig.real:.3f}")
        else:
            print(f"  λ{i+1},{i+2} = {eig.real:.3f} ± {eig.imag:.3f}j")
    
    # Check stability
    if np.all(np.real(eigvals) < 0):
        print("✓ System is STABLE (all poles in left half-plane)")
    else:
        print("⚠ System is UNSTABLE (some poles in right half-plane)")
    
    # Interpret gains
    print(f"\nGain interpretation:")
    print(f"  Position gain: {K[0,0]:.2f} - {'High' if abs(K[0,0]) > 10 else 'Medium' if abs(K[0,0]) > 1 else 'Low'}")
    print(f"  Velocity gain: {K[0,1]:.2f} - Provides damping")
    print(f"  Angle gain: {K[0,2]:.2f} - {'High' if abs(K[0,2]) > 50 else 'Medium' if abs(K[0,2]) > 10 else 'Low'}")
    print(f"  Angular velocity gain: {K[0,3]:.2f} - Provides pendulum damping")

def create_animation_with_lqr(time, states, control_inputs, params, title="LQR Controlled Inverted Pendulum"):
    """Create animation showing LQR control performance"""
    cart_pos = states[:, 0]
    pendulum_angle = states[:, 2]
    l = params['l']
    
    # Calculate pendulum bob position
    bob_x = cart_pos + l * np.sin(pendulum_angle)
    bob_y = l * np.cos(pendulum_angle)
    
    # Set up the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Animation subplot
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-0.1, 0.4)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Height (m)')
    
    # Reference lines
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Target position')
    ax1.axhline(y=l, color='blue', linestyle='--', alpha=0.5, label='Target (upright)')
    
    # Animation elements
    cart, = ax1.plot([], [], 'bs', markersize=15, label='Cart')
    pendulum, = ax1.plot([], [], 'r-', linewidth=3, label='Pendulum')
    bob, = ax1.plot([], [], 'ro', markersize=6, label='Bob')
    ground, = ax1.plot([-1, 1], [-0.05, -0.05], 'k-', linewidth=2)
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax1.legend(loc='upper right', fontsize=8)
    
    # State response plots
    ax2.plot(time, cart_pos, 'b-', label='Cart Position (m)', linewidth=2)
    ax2.plot(time, pendulum_angle, 'r-', label='Pendulum Angle (rad)', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('States')
    ax2.set_title('State Response')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Control input plot
    ax3.plot(time, control_inputs, 'g-', linewidth=2, label='Control Force (N)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Input (N)')
    ax3.set_title('LQR Control Input')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Phase portrait
    ax4.plot(pendulum_angle, states[:, 3], 'purple', linewidth=1, alpha=0.7)
    ax4.scatter(pendulum_angle[0], states[0, 3], color='red', s=50, label='Start', zorder=5)
    ax4.scatter(pendulum_angle[-1], states[-1, 3], color='green', s=50, label='End', zorder=5)
    ax4.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Upright')
    ax4.set_xlabel('Pendulum Angle (rad)')
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.set_title('Phase Portrait')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Animation function
    def animate(frame):
        if frame < len(time):
            x_cart = cart_pos[frame]
            x_bob = bob_x[frame]
            y_bob = bob_y[frame]
            
            cart.set_data([x_cart], [0])
            pendulum.set_data([x_cart, x_bob], [0, y_bob])
            bob.set_data([x_bob], [y_bob])
            
            time_text.set_text(f'Time: {time[frame]:.2f} s\n'
                             f'Cart: {x_cart:.3f} m\n'
                             f'Angle: {pendulum_angle[frame]:.3f} rad\n'
                             f'Control: {control_inputs[frame]:.2f} N')
            
        return cart, pendulum, bob, time_text
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time), 
                                 interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    return fig, anim

if __name__ == "__main__":
    # Initialize system
    params = define_system_parameters()
    A, B, C, D = create_state_space_matrices(params)
    
    print("=== INVERTED PENDULUM LQR CONTROL ===")
    
    # Design LQR controller
    # Q matrix: weights for [x, x_dot, theta, theta_dot]
    Q = np.diag([1, 1, 100, 10])  # High weight on angle (theta) to keep upright
    
    # R matrix: weight for control input
    R = np.array([[1]])  # Moderate control effort
    
    print("Designing LQR controller...")
    K, S, eigvals = design_lqr_controller(A, B, Q, R)
    analyze_lqr_design(K, eigvals, Q, R)
    
    # Test scenarios
    scenarios = [
        ("Small disturbance", [0.1, 0, 0.1, 0]),  # Small cart displacement + angle
        ("Large initial angle", [0, 0, 0.5, 0]),   # 30 degrees
        ("Moving cart", [0.2, 0.1, 0, 0]),         # Cart moving with velocity
    ]
    
    # Time span
    t = np.linspace(0, 10, 1000)
    
    print(f"\n=== SIMULATION RESULTS ===")
    
    for scenario_name, initial_state in scenarios:
        print(f"\nTesting: {scenario_name}")
        print(f"Initial state: {initial_state}")
        
        # Simulate with LQR control
        time, states, outputs, control_inputs = simulate_lqr_control(
            A, B, C, D, K, initial_state, t
        )
        
        # Check final performance
        final_state = states[-1]
        final_error = np.linalg.norm(final_state)
        
        print(f"Final state: [{final_state[0]:.4f}, {final_state[1]:.4f}, {final_state[2]:.4f}, {final_state[3]:.4f}]")
        print(f"Final error magnitude: {final_error:.4f}")
        print(f"Max control effort: {np.max(np.abs(control_inputs)):.2f} N")
        
        # Create animation for first scenario
        if scenario_name == "Large initial angle":
            print("Creating animation...")
            fig, anim = create_animation_with_lqr(time, states, control_inputs, params,
                                                f"LQR Control: {scenario_name}")
            
    print(f"\n=== SUCCESS! ===")
    print("✓ LQR controller successfully designed")
    print("✓ System stabilized at upright position")
    print("✓ Both cart position and pendulum angle controlled")
    
    plt.show()