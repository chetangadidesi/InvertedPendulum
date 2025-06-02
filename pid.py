# INVERTED PENDULUM

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
from scipy import signal


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, output_limits= None):
        """PID Controller for inverted pendulum

        Args:
            Kp : Proportional Gain
            Ki : Integral Gain
            Kd : Derivative Gain
            dt : Time Step
            output_limits : Tuple (min,max) for output saturation
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits
        
        #initialize PID terms
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error):
        """Update PID controller with current error

        Args:
            error : Current error ( Reference - Measurement )
        """
        # Proportional Error
        P = self.Kp * error 
        # Integral Error
        self.integral += error*self.dt
        I = self.Ki * self.integral
        # Derivative Error
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative
        
        # PID output
        output = P + I + D
        
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
            
            # Anti-windup: prevent integral windup when saturated
            if (output >= self.output_limits[1] and error > 0) or \
               (output <= self.output_limits[0] and error < 0):
                self.integral -= error * self.dt  # Remove the current error from integral
        
        # Store current error for next iteration
        self.prev_error = error
        
        return output
    
    def reset(self):
        # Reset PID Controller
        self.prev_error = 0.0
        self.integral = 0.0
    


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


def state_space_dynamics_pid(x, t, pid_controller, reference, A, B):
    """
    Modified state space dynamics function for PID control
    
    Args:
        x : state vector [cart_position, cart_velocity, pendulum_angle, pendulum_velocity]
        t : time 
        pid_controller : PID controller object
        reference : desired pendulum angle (usually 0)
        A : state space matrix
        B : state space matrix
    
    Returns:
        dx_dt : state derivative
    """
    x = np.array(x).reshape(-1, 1)
    
    # Extract pendulum angle (3rd state)
    current_angle = x[2, 0]
    
    # Calculate error
    error = reference - current_angle
    
    # Get PID control output
    u = pid_controller.update(error)
    
    # Apply control input
    u_vec = np.array([[u]])
    dx_dt = A @ x + B @ u_vec
    
    return dx_dt.flatten()

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


def simulate_closed_loop_pid(A, B, C, D, initial_state, time_span, pid_controller, reference=0):
    """
    Simulate the system with closed-loop PID control
    
    Args:
        A : state matrix
        B : state matrix  
        C : state matrix
        D : state matrix
        initial_state : [x0, x_dot0, theta0, theta_dot0]
        time_span : array of time points
        pid_controller : PID controller object
        reference : desired pendulum angle (default: 0 for upright)
        
    Returns:
        t : time array
        states : state history
        outputs : output history
        control_inputs : history of control inputs
    """
    # Reset PID controller
    pid_controller.reset()
    
    # Initialize arrays to store results
    n_points = len(time_span)
    states = np.zeros((n_points, 4))  # 4 states
    control_inputs = np.zeros(n_points)
    outputs = np.zeros((n_points, 2))  # 2 outputs
    
    # Set initial conditions
    states[0] = initial_state
    
    # Calculate initial control input
    current_angle = initial_state[2]
    error = reference - current_angle
    control_inputs[0] = pid_controller.update(error)
    
    # Calculate initial output
    state_vec = np.array(initial_state).reshape(-1, 1)
    u_vec = np.array([[control_inputs[0]]])
    outputs[0] = (C @ state_vec + D @ u_vec).flatten()
    
    # Simulate step by step
    dt = time_span[1] - time_span[0]
    
    for i in range(1, n_points):
        # Get previous state
        prev_state = states[i-1]
        
        # Calculate control input based on current pendulum angle
        current_angle = prev_state[2]
        error = reference - current_angle
        u = pid_controller.update(error)
        control_inputs[i] = u
        
        # Integrate one step using Euler method (more stable than odeint for this)
        # dx/dt = Ax + Bu
        x_vec = prev_state.reshape(-1, 1)
        u_vec = np.array([[u]])
        dx_dt = (A @ x_vec + B @ u_vec).flatten()
        
        # Update state: x(k+1) = x(k) + dt * dx/dt
        states[i] = prev_state + dt * dx_dt
        
        # Calculate output
        state_vec = states[i].reshape(-1, 1)
        outputs[i] = (C @ state_vec + D @ u_vec).flatten()
    
    return time_span, states, outputs, control_inputs


def simulate_impulse_response_pid(A, B, C, D, time_span, pid_controller, impulse_magnitude=1.0, reference=0):
    """
    Simulate the system's impulse response with closed-loop PID control
    
    Args:
        A : state matrix
        B : state matrix  
        C : state matrix
        D : state matrix
        time_span : array of time points
        pid_controller : PID controller object
        impulse_magnitude : magnitude of the impulse disturbance
        reference : desired pendulum angle (default: 0 for upright)
        
    Returns:
        t : time array
        states : state history
        outputs : output history
        control_inputs : history of control inputs
    """
    # Reset PID controller
    pid_controller.reset()
    
    # Initialize arrays to store results
    n_points = len(time_span)
    states = np.zeros((n_points, 4))  # 4 states
    control_inputs = np.zeros(n_points)
    outputs = np.zeros((n_points, 2))  # 2 outputs
    
    # Set initial conditions (start at equilibrium)
    initial_state = [0, 0, 0, 0]  # All states start at zero
    states[0] = initial_state
    
    # Calculate initial control input
    current_angle = initial_state[2]
    error = reference - current_angle
    control_inputs[0] = pid_controller.update(error)
    
    # Calculate initial output
    state_vec = np.array(initial_state).reshape(-1, 1)
    u_vec = np.array([[control_inputs[0]]])
    outputs[0] = (C @ state_vec + D @ u_vec).flatten()
    
    # Simulate step by step
    dt = time_span[1] - time_span[0]
    
    for i in range(1, n_points):
        # Get previous state
        prev_state = states[i-1]
        
        # Calculate control input based on current pendulum angle
        current_angle = prev_state[2]
        error = reference - current_angle
        u_pid = pid_controller.update(error)
        
        # Add impulse disturbance at the first time step
        if i == 1:  # Apply impulse at t = dt (first step after initial)
            u_total = u_pid + impulse_magnitude / dt  # Impulse = magnitude/dt
        else:
            u_total = u_pid
            
        control_inputs[i] = u_total
        
        # Integrate one step using Euler method
        # dx/dt = Ax + Bu
        x_vec = prev_state.reshape(-1, 1)
        u_vec = np.array([[u_total]])
        dx_dt = (A @ x_vec + B @ u_vec).flatten()
        
        # Update state: x(k+1) = x(k) + dt * dx/dt
        states[i] = prev_state + dt * dx_dt
        
        # Calculate output
        state_vec = states[i].reshape(-1, 1)
        u_output_vec = np.array([[u_pid]])  # For output calculation, use only PID part
        outputs[i] = (C @ state_vec + D @ u_output_vec).flatten()
    
    return time_span, states, outputs, control_inputs


def create_pendulum_animation(time, states, params, save_gif=True, filename='pendulum.gif'):
    """
    Create an animated visualization of the inverted pendulum system
    """
    
    # Extract data
    cart_positions = states[:, 0]  # Cart position
    pendulum_angles = states[:, 2]  # Pendulum angle
    
    # System parameters
    L = params['l']  # Pendulum length
    
    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Animation subplot
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.2, 0.8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Inverted Pendulum Animation (PID Control)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Height (m)')
    
    # Plot elements for animation
    track_line, = ax1.plot([-1.5, 1.5], [0, 0], 'k-', linewidth=4, label='Track')
    cart_patch = plt.Rectangle((0, 0), 0.15, 0.08, fc='blue', alpha=0.8)
    ax1.add_patch(cart_patch)
    
    pendulum_line, = ax1.plot([], [], 'ro-', linewidth=3, markersize=8, label='Pendulum')
    pendulum_mass, = ax1.plot([], [], 'ro', markersize=12)
    
    # Trace line for pendulum tip
    trace_line, = ax1.plot([], [], 'r--', alpha=0.5, linewidth=1)
    trace_x, trace_y = [], []
    
    # Text displays
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    angle_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    pos_text = ax1.text(0.02, 0.75, '', transform=ax1.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax1.legend(loc='upper right')
    
    # Data plots subplot
    ax2.plot(time, pendulum_angles, 'r-', label='Pendulum Angle (rad)', linewidth=2)
    ax2.plot(time, cart_positions, 'b-', label='Cart Position (m)', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position/Angle')
    ax2.set_title('System Response', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Progress indicator
    progress_line = ax2.axvline(x=0, color='green', linewidth=2, alpha=0.7)
    
    def animate(frame):
        # Current simulation data
        current_time = time[frame]
        cart_pos = cart_positions[frame]
        pend_angle = pendulum_angles[frame]
        
        # Calculate pendulum tip position
        # Note: angle is measured from vertical (0 = upright)
        pend_x = cart_pos + L * np.sin(pend_angle)
        pend_y = L * np.cos(pend_angle)
        
        # Update cart position (centered rectangle)
        cart_patch.set_x(cart_pos - 0.075)  # Center the cart
        cart_patch.set_y(-0.04)
        
        # Update pendulum line (from cart to pendulum mass)
        pendulum_line.set_data([cart_pos, pend_x], [0, pend_y])
        
        # Update pendulum mass position
        pendulum_mass.set_data([pend_x], [pend_y])
        
        # Update trace (show path of pendulum tip)
        trace_x.append(pend_x)
        trace_y.append(pend_y)
        
        # Keep only last 150 points for trace
        if len(trace_x) > 150:
            trace_x.pop(0)
            trace_y.pop(0)
        
        trace_line.set_data(trace_x, trace_y)
        
        # Update text displays
        time_text.set_text(f'Time: {current_time:.2f} s')
        angle_text.set_text(f'Angle: {np.degrees(pend_angle):.1f}°')
        pos_text.set_text(f'Cart Pos: {cart_pos:.3f} m')
        
        # Update progress indicator
        progress_line.set_xdata([current_time, current_time])
        
        return (cart_patch, pendulum_line, pendulum_mass, trace_line, 
                time_text, angle_text, pos_text, progress_line)
    
    # Create animation
    # Skip frames to make animation smoother
    skip_frames = max(1, len(time) // 400)  # Limit to ~400 frames max
    frames = range(0, len(time), skip_frames)
    
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                 interval=30, blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Save as GIF if requested
    if save_gif:
        print(f"Saving animation as {filename}...")
        try:
            anim.save(filename, writer='pillow', fps=25)
            print("Animation saved!")
        except:
            print("Could not save GIF. Make sure pillow is installed: pip install pillow")
    
    return fig, anim
    


if __name__ == "__main__":
    #Initialize system
    params = define_system_parameters()
    A, B, C, D = create_state_space_matrices(params)
    
    #Print matrices to verify
    print_state_matrices(A, B, C, D, params)
    
    dt = 0.01
    pid = PIDController(Kp=100, Ki=1, Kd=5, dt=dt, output_limits=(-100, 100))
    
    # Time vector matching MATLAB (0:0.01:10)
    t = np.arange(0, 10, dt)
    
    # Simulate impulse response
    time, states, outputs, control_inputs = simulate_impulse_response_pid(
        A, B, C, D, t, pid, impulse_magnitude=1, reference=0)

    # Plot impulse response - matching MATLAB plot
    plt.figure(figsize=(12, 8))
    
    # Main plot - Pendulum angle response to impulse
    plt.subplot(2, 2, 1)
    plt.plot(time, states[:, 2])  # Pendulum angle
    plt.xlim([0, 2.5])
    plt.ylim([-0.2, 0.2])
    plt.title('Response of Pendulum Position to an Impulse Disturbance\nunder PID Control: Kp = 100, Ki = 1, Kd = 5')
    plt.xlabel('Time (s)')
    plt.ylabel('Pendulum Angle (rad)')
    plt.grid(True)
    
    # Additional plots for analysis
    plt.subplot(2, 2, 2)
    plt.plot(time, states[:, 0])  # Cart position
    plt.title('Cart Position Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(time, control_inputs)  # Control effort
    plt.title('Control Input (Force)')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(time, states[:, 2], label='Pendulum Angle')
    plt.plot(time, states[:, 0], label='Cart Position')
    plt.title('System Response Overview')
    plt.xlabel('Time (s)')
    plt.ylabel('Position/Angle')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    # NEW: Create and show animation
    print("\nCreating animation...")
    fig_anim, anim = create_pendulum_animation(time, states, params, 
                                             save_gif=False,  # Set to True to save GIF
                                             filename='inverted_pendulum_control.gif')
    
    print("Animation created! Close the plot window when done viewing.")
    plt.show()
    
    # Performance analysis
    print(f"\nControl Performance Analysis:")
    print(f"Peak pendulum angle: {np.max(np.abs(states[:, 2])):.4f} rad ({np.degrees(np.max(np.abs(states[:, 2]))):.1f}°)")
    
    # Print some key performance metrics
    print(f"\nImpulse Response Analysis:")
    print(f"Peak pendulum angle: {np.max(np.abs(states[:, 2])):.4f} rad")
    print(f"Settling time (2% criterion): {time[np.where(np.abs(states[:, 2]) < 0.02*np.max(np.abs(states[:, 2])))[0][-1]] if len(np.where(np.abs(states[:, 2]) < 0.02*np.max(np.abs(states[:, 2])))[0]) > 0 else 'Not settled':.2f} s")
    print(f"Final cart position: {states[-1, 0]:.4f} m")
    
    
    
    
    
    
    """
    # Simulate closed-loop
    initial_state = [0, 0, 0.1, 0]  # Small initial angle disturbance
    t = np.arange(0, 5, dt)

    time, states, outputs, control_inputs = simulate_closed_loop_pid(
        A, B, C, D, initial_state, t, pid, reference=0)

    # Plot results including control effort
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(time, states[:, 2])  # Pendulum angle
    plt.title('Pendulum Angle (PID Control)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time, states[:, 0])  # Cart position
    plt.title('Cart Position')
    plt.ylabel('Position (m)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time, control_inputs)  # Control effort
    plt.title('Control Input (Force)')
    plt.ylabel('Force (N)')
    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time, states[:, 2], label='Pendulum Angle')
    plt.axhline(y=0, color='r', linestyle='--', label='Reference')
    plt.title('Tracking Performance')
    plt.ylabel('Angle (rad)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    
"""