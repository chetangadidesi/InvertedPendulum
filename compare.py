import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
from scipy import signal
from scipy.linalg import solve_continuous_are, inv


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, output_limits=None):
        """PID Controller for inverted pendulum"""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits
        
        # Initialize PID terms
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error):
        """Update PID controller with current error"""
        # Proportional Error
        P = self.Kp * error 
        # Integral Error
        self.integral += error * self.dt
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
                self.integral -= error * self.dt
        
        # Store current error for next iteration
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset PID Controller"""
        self.prev_error = 0.0
        self.integral = 0.0


def define_system_parameters():
    params = {
        'M': 0.5,   # Mass of cart
        'm': 0.2,   # Mass of pendulum
        'b': 0.1,   # Coeff of friction
        'l': 0.3,   # Length of pendulum
        'I': 0.006, # Moment of Inertia
        'g': 9.81   # gravity
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
    """Design LQR controller for the system"""
    # Solve the continuous-time algebraic Riccati equation (CARE)
    S = solve_continuous_are(A, B, Q, R)
    
    # Calculate the LQR gain matrix K
    K = inv(R) @ B.T @ S
    
    # Calculate closed-loop eigenvalues
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)
    
    return K, S, eigvals


def lqr_control(state, K, reference=[0, 0, 0, 0]):
    """Calculate LQR control input"""
    state = np.array(state).reshape(-1, 1)
    ref = np.array(reference).reshape(-1, 1)
    
    # Control law: u = -K * (x - x_ref)
    u = -K @ (state - ref)
    
    return u[0, 0]  # Return scalar


def simulate_pid_control(A, B, C, D, initial_state, time_span, pid_controller, reference=0):
    """Simulate the system with PID control"""
    pid_controller.reset()
    
    n_points = len(time_span)
    states = np.zeros((n_points, 4))
    control_inputs = np.zeros(n_points)
    
    states[0] = initial_state
    
    # Calculate initial control input
    current_angle = initial_state[2]
    error = reference - current_angle
    control_inputs[0] = pid_controller.update(error)
    
    dt = time_span[1] - time_span[0]
    
    for i in range(1, n_points):
        prev_state = states[i-1]
        
        # Calculate control input based on current pendulum angle
        current_angle = prev_state[2]
        error = reference - current_angle
        u = pid_controller.update(error)
        control_inputs[i] = u
        
        # Integrate one step using Euler method
        x_vec = prev_state.reshape(-1, 1)
        u_vec = np.array([[u]])
        dx_dt = (A @ x_vec + B @ u_vec).flatten()
        
        # Update state
        states[i] = prev_state + dt * dx_dt
    
    return time_span, states, control_inputs


def simulate_lqr_control(A, B, C, D, K, initial_state, time_span, reference=[0, 0, 0, 0]):
    """Simulate the system with LQR control"""
    states = []
    control_inputs = []
    
    # Simulate step by step
    dt = time_span[1] - time_span[0]
    current_state = initial_state
    
    for t in time_span:
        states.append(current_state.copy())
        
        # Calculate control input
        u_lqr = lqr_control(current_state, K, reference)
        control_inputs.append(u_lqr)
        
        # Integrate one step forward
        if t < time_span[-1]:
            next_states = odeint(lambda x, t: (A @ np.array(x).reshape(-1,1) + B * u_lqr).flatten(), 
                               current_state, [t, t + dt])
            current_state = next_states[1]
    
    states = np.array(states)
    control_inputs = np.array(control_inputs)
    
    return time_span, states, control_inputs


def create_comparison_animation(time, states_pid, control_pid, states_lqr, control_lqr, params, filename='compare.gif'):
    """Create side-by-side animation comparing PID and LQR control"""
    
    # Extract data
    cart_pos_pid = states_pid[:, 0]
    pend_angle_pid = states_pid[:, 2]
    cart_pos_lqr = states_lqr[:, 0]
    pend_angle_lqr = states_lqr[:, 2]
    
    l = params['l']  # Pendulum length
    
    # Calculate pendulum bob positions
    bob_x_pid = cart_pos_pid + l * np.sin(pend_angle_pid)
    bob_y_pid = l * np.cos(pend_angle_pid)
    bob_x_lqr = cart_pos_lqr + l * np.sin(pend_angle_lqr)
    bob_y_lqr = l * np.cos(pend_angle_lqr)
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Animation subplots (top row)
    ax_pid = fig.add_subplot(gs[0, 0:2])
    ax_lqr = fig.add_subplot(gs[0, 2:4])
    
    # Performance comparison plots (bottom rows)
    ax_states = fig.add_subplot(gs[1, :])
    ax_control = fig.add_subplot(gs[2, :])
    
    # Set up animation axes
    for ax, title in [(ax_pid, 'PID Control'), (ax_lqr, 'LQR Control')]:
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.1, 0.4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        
        # Reference lines
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=l, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        
        # Ground line
        ax.plot([-0.8, 0.8], [-0.05, -0.05], 'k-', linewidth=2)
    
    # Animation elements for PID
    cart_pid = plt.Rectangle((0, 0), 0.1, 0.06, fc='blue', alpha=0.8)
    ax_pid.add_patch(cart_pid)
    pendulum_line_pid, = ax_pid.plot([], [], 'ro-', linewidth=3, markersize=8)
    trace_line_pid, = ax_pid.plot([], [], 'r--', alpha=0.5, linewidth=1)
    
    # Animation elements for LQR
    cart_lqr = plt.Rectangle((0, 0), 0.1, 0.06, fc='red', alpha=0.8)
    ax_lqr.add_patch(cart_lqr)
    pendulum_line_lqr, = ax_lqr.plot([], [], 'go-', linewidth=3, markersize=8)
    trace_line_lqr, = ax_lqr.plot([], [], 'g--', alpha=0.5, linewidth=1)
    
    # Text displays
    text_pid = ax_pid.text(0.02, 0.95, '', transform=ax_pid.transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    text_lqr = ax_lqr.text(0.02, 0.95, '', transform=ax_lqr.transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    # Performance comparison plots
    ax_states.plot(time, pend_angle_pid, 'b-', label='PID - Pendulum Angle', linewidth=2)
    ax_states.plot(time, pend_angle_lqr, 'r-', label='LQR - Pendulum Angle', linewidth=2)
    ax_states.plot(time, cart_pos_pid, 'b--', label='PID - Cart Position', linewidth=1.5, alpha=0.7)
    ax_states.plot(time, cart_pos_lqr, 'r--', label='LQR - Cart Position', linewidth=1.5, alpha=0.7)
    ax_states.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_states.set_xlabel('Time (s)')
    ax_states.set_ylabel('States')
    ax_states.set_title('State Response Comparison')
    ax_states.legend(ncol=2, loc='upper right')
    ax_states.grid(True, alpha=0.3)
    
    ax_control.plot(time, control_pid, 'b-', label='PID Control Input', linewidth=2)
    ax_control.plot(time, control_lqr, 'r-', label='LQR Control Input', linewidth=2)
    ax_control.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_control.set_xlabel('Time (s)')
    ax_control.set_ylabel('Control Force (N)')
    ax_control.set_title('Control Input Comparison')
    ax_control.legend()
    ax_control.grid(True, alpha=0.3)
    
    # Progress indicators
    progress_line_states = ax_states.axvline(x=0, color='purple', linewidth=2, alpha=0.7)
    progress_line_control = ax_control.axvline(x=0, color='purple', linewidth=2, alpha=0.7)
    
    # Trace storage
    trace_x_pid, trace_y_pid = [], []
    trace_x_lqr, trace_y_lqr = [], []
    
    def animate(frame):
        current_time = time[frame]
        
        # PID animation
        cart_x_pid = cart_pos_pid[frame]
        pend_x_pid = bob_x_pid[frame]
        pend_y_pid = bob_y_pid[frame]
        
        cart_pid.set_x(cart_x_pid - 0.05)
        cart_pid.set_y(-0.03)
        pendulum_line_pid.set_data([cart_x_pid, pend_x_pid], [0, pend_y_pid])
        
        # LQR animation
        cart_x_lqr = cart_pos_lqr[frame]
        pend_x_lqr = bob_x_lqr[frame]
        pend_y_lqr = bob_y_lqr[frame]
        
        cart_lqr.set_x(cart_x_lqr - 0.05)
        cart_lqr.set_y(-0.03)
        pendulum_line_lqr.set_data([cart_x_lqr, pend_x_lqr], [0, pend_y_lqr])
        
        # Update traces
        trace_x_pid.append(pend_x_pid)
        trace_y_pid.append(pend_y_pid)
        trace_x_lqr.append(pend_x_lqr)
        trace_y_lqr.append(pend_y_lqr)
        
        # Keep trace length manageable
        if len(trace_x_pid) > 100:
            trace_x_pid.pop(0)
            trace_y_pid.pop(0)
            trace_x_lqr.pop(0)
            trace_y_lqr.pop(0)
        
        trace_line_pid.set_data(trace_x_pid, trace_y_pid)
        trace_line_lqr.set_data(trace_x_lqr, trace_y_lqr)
        
        # Update text displays
        text_pid.set_text(f'Time: {current_time:.2f} s\n'
                         f'Angle: {np.degrees(pend_angle_pid[frame]):.1f}°\n'
                         f'Cart: {cart_x_pid:.3f} m\n'
                         f'Force: {control_pid[frame]:.2f} N')
        
        text_lqr.set_text(f'Time: {current_time:.2f} s\n'
                         f'Angle: {np.degrees(pend_angle_lqr[frame]):.1f}°\n'
                         f'Cart: {cart_x_lqr:.3f} m\n'
                         f'Force: {control_lqr[frame]:.2f} N')
        
        # Update progress indicators
        progress_line_states.set_xdata([current_time, current_time])
        progress_line_control.set_xdata([current_time, current_time])
        
        return (cart_pid, pendulum_line_pid, trace_line_pid, cart_lqr, 
                pendulum_line_lqr, trace_line_lqr, text_pid, text_lqr, 
                progress_line_states, progress_line_control)
    
    # Create animation
    frames = range(0, len(time), max(1, len(time) // 500))  # Limit frames for smooth animation
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                 interval=40, blit=False, repeat=True)
    #anim.save(filename, writer='pillow', fps=60)
    #print("Animation saved!")
    
    return fig, anim


def print_performance_comparison(time, states_pid, control_pid, states_lqr, control_lqr):
    """Print quantitative performance comparison"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: PID vs LQR")
    print("="*60)
    
    # Calculate performance metrics
    def calculate_metrics(states, control):
        cart_pos = states[:, 0]
        pend_angle = states[:, 2]
        
        # Settling time (2% criterion for pendulum angle)
        angle_envelope = 0.02 * np.max(np.abs(pend_angle))
        settled_indices = np.where(np.abs(pend_angle) <= angle_envelope)[0]
        settling_time = time[settled_indices[-1]] if len(settled_indices) > 0 else "Not settled"
        
        # Performance metrics
        max_angle = np.max(np.abs(pend_angle))
        max_cart_pos = np.max(np.abs(cart_pos))
        max_control = np.max(np.abs(control))
        rms_control = np.sqrt(np.mean(control**2))
        final_angle_error = np.abs(pend_angle[-1])
        final_cart_error = np.abs(cart_pos[-1])
        
        return {
            'settling_time': settling_time,
            'max_angle': max_angle,
            'max_cart_pos': max_cart_pos,
            'max_control': max_control,
            'rms_control': rms_control,
            'final_angle_error': final_angle_error,
            'final_cart_error': final_cart_error
        }
    
    metrics_pid = calculate_metrics(states_pid, control_pid)
    metrics_lqr = calculate_metrics(states_lqr, control_lqr)
    
    # Print comparison table
    print(f"{'Metric':<25} {'PID':<15} {'LQR':<15} {'Winner':<10}")
    print("-" * 65)
    
    comparisons = [
        ('Max Pendulum Angle', f"{metrics_pid['max_angle']:.4f} rad", f"{metrics_lqr['max_angle']:.4f} rad", 
         'PID' if metrics_pid['max_angle'] < metrics_lqr['max_angle'] else 'LQR'),
        ('Max Cart Position', f"{metrics_pid['max_cart_pos']:.4f} m", f"{metrics_lqr['max_cart_pos']:.4f} m",
         'PID' if metrics_pid['max_cart_pos'] < metrics_lqr['max_cart_pos'] else 'LQR'),
        ('Max Control Force', f"{metrics_pid['max_control']:.2f} N", f"{metrics_lqr['max_control']:.2f} N",
         'PID' if metrics_pid['max_control'] < metrics_lqr['max_control'] else 'LQR'),
        ('RMS Control Effort', f"{metrics_pid['rms_control']:.2f} N", f"{metrics_lqr['rms_control']:.2f} N",
         'PID' if metrics_pid['rms_control'] < metrics_lqr['rms_control'] else 'LQR'),
        ('Final Angle Error', f"{metrics_pid['final_angle_error']:.5f} rad", f"{metrics_lqr['final_angle_error']:.5f} rad",
         'PID' if metrics_pid['final_angle_error'] < metrics_lqr['final_angle_error'] else 'LQR'),
        ('Final Cart Error', f"{metrics_pid['final_cart_error']:.5f} m", f"{metrics_lqr['final_cart_error']:.5f} m",
         'PID' if metrics_pid['final_cart_error'] < metrics_lqr['final_cart_error'] else 'LQR'),
    ]
    
    for metric, pid_val, lqr_val, winner in comparisons:
        print(f"{metric:<25} {pid_val:<15} {lqr_val:<15} {winner:<10}")


if __name__ == "__main__":
    print("INVERTED PENDULUM CONTROL COMPARISON: PID vs LQR")
    print("="*55)
    
    # Initialize system
    params = define_system_parameters()
    A, B, C, D = create_state_space_matrices(params)
    
    # Set up controllers
    dt = 0.01
    pid_controller = PIDController(Kp=100, Ki=1, Kd=2, dt=dt, output_limits=(-100, 100))
    
    # LQR design
    Q = np.diag([1, 1, 100, 10])  # State weights
    R = np.array([[1]])           # Control weight
    K, S, eigvals = design_lqr_controller(A, B, Q, R)
    
    print(f"PID Parameters: Kp={pid_controller.Kp}, Ki={pid_controller.Ki}, Kd={pid_controller.Kd}")
    print(f"LQR Gains: K = {K}")
    
    # Simulation parameters
    initial_state = [0, 0, 0.3, 0]  # Start with 0.2 rad (~11.5°) angle disturbance
    t = np.arange(0, 8, dt)
    
    print(f"\nSimulating with initial state: {initial_state}")
    print(f"Initial pendulum angle: {np.degrees(initial_state[2]):.1f}°")
    
    # Run simulations
    print("\nRunning PID simulation...")
    time_pid, states_pid, control_pid = simulate_pid_control(
        A, B, C, D, initial_state, t, pid_controller, reference=0)
    
    print("Running LQR simulation...")
    time_lqr, states_lqr, control_lqr = simulate_lqr_control(
        A, B, C, D, K, initial_state, t, reference=[0, 0, 0, 0])
    
    # Print performance comparison
    print_performance_comparison(t, states_pid, control_pid, states_lqr, control_lqr)
    
    # Create comparison animation
    print("\nCreating comparison animation...")
    fig, anim = create_comparison_animation(t, states_pid, control_pid, 
                                          states_lqr, control_lqr, params)
    
    plt.suptitle('Inverted Pendulum Control Comparison: PID vs LQR', fontsize=16, fontweight='bold')
    
    print("\n" + "="*60)
    print("ANIMATION READY!")
    print("• Blue cart/pendulum: PID Control")
    print("• Red cart/pendulum: LQR Control") 
    print("• Dashed lines show pendulum tip traces")
    print("• Bottom plots show quantitative comparison")
    print("="*60)
    
    plt.show()