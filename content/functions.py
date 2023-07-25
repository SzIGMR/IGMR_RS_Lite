""" Main library for the functions needed by the scripts. """
__docformat__ = "numpy"


# Import the Stuff
from dataclasses import dataclass
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Helper Functions
def comp_omega(stiffness, mass):
    r"""Computes omega:
    omega_0 = sqrt(c/m)

    Parameters
    ----------
    `stiffness`
        Stiffness of the System
    `mass`
        Mass of the System

    Returns
    -------
    omega_0
        Eigenkreisfreqency
    """
    return np.sqrt(stiffness / mass)


def comp_delta(damping, mass):
    """Computes delta:
    delta = k/(2 * mass)

    Parameters
    ----------
    `damping`
        Damping of System
    `mass`
        Mass of the System

    Returns
    -------
    delta
        Decay constant
    """
    return damping / (2 * mass)


def comp_solution(
        parameter,
        solution_type
        ):
    """ Helper function for switching between numerical and analytical solution.
    Parameters are getting forwarded to proper solutions.

    Parameters
    ----------
    `parameter`
        Dictionary with parameters as required by `comp_solution_analytical`
    `solution_type` : str
        One off numerical or analytical
    """
    if solution_type == "numerical":
        return comp_solution_numerical(parameter)
    if solution_type == "analytical":
        return comp_solution_analytical(parameter)

    print(f"Wrong solution_type selected, {solution_type} is not a valid choice.")


def comp_solution_numerical(parameter):
    """ Computing of numerical solution.
    Also switches between the m,c and k type vs. delta and omega.

    Parameters
    ----------
    `parameter`
        Dictionary with parameters as required by `comp_solution_numerical`

    Returns
    -------
    [sol.y[1]] : np.array
        Time-series of displacement.
    """
    mass = parameter.get('mass')
    stiffness = parameter.get('stiffness')
    damping = parameter.get('damping')
    x_0 = parameter.get('x_0')
    x_dot_0 = parameter.get('x_dot_0')
    time = parameter.get('time')

    delta = parameter.get('delta')
    omega_0 = parameter.get('omega_0')

    initial_values = [x_dot_0, x_0]

    if None not in (omega_0, delta):
        sol = solve_ivp(state_space_representation_omega, y0=initial_values,
                        t_span=[0, time[-1]], t_eval=time,
                        args=(delta, omega_0))
    else:
        sol = solve_ivp(state_space_representation, y0=initial_values,
                        t_span=[0, time[-1]], t_eval=time,
                        args=(mass, stiffness, damping))
    return [sol.y[1]]


def state_space_representation(t, u, m, c, k):
    """ State space representation with respect to physical
    u = (v, -c/m * x - k/m * v)

    Parameters
    ----------
    `t`
        Time of solution vector
    `u`
        State space vector [v, x]
    `m`
        Mass
    `c`
        Stiffness
    `k`
        Damping

    Returns
    -------
    np.array
        State Space representation as 2d vector

    """
    xdot, x = u
    return [- c/m * x - k/m * xdot, xdot]


def state_space_representation_omega(t, u, delta, omega):
    """ State space representation with respect to physical
    u = (v, - omega^2 * x - 2 *delta * v)

    Parameters
    ----------
    `t`
        Time of solution vector
    `u`
        State space vector [v, x]
    `delta`
       Decay constant
    `omega`
       Eigenfrequency

    Returns
    -------
    np.array
        State Space representation as 2d vector

    """
    xdot, x = u
    return [- np.power(omega, 2) * x - 2 * delta * xdot, xdot]


def comp_solution_analytical(parameter):
    """
    Parameters
    ----------
    `parameter`
        Dictionary with parameters such as mass,
        stiffness, damping, x_0, x_dot_0 and time.
        Further if delta, omega_0, f_hat, Omega are
        supplied the solution is calculated over them.

    Returns
    -------
    x_h + x_p : np.array
        x = (x_h + x_p) Analytical solution of the one-degree system
    x_h : np.array
        harmonic solution part
    x_p : np.array
        particular solution part
    """
    mass = parameter.get('mass')
    stiffness = parameter.get('stiffness')
    damping = parameter.get('damping')
    x_0 = parameter.get('x_0')
    x_dot_0 = parameter.get('x_dot_0')
    time = parameter.get('time')

    delta = parameter.get('delta')
    omega_0 = parameter.get('omega_0')
    f_hat = parameter.get('f_hat')
    Omega = parameter.get('Omega')

    # calculate constants
    if delta is None:
        if mass > 0:
            delta = comp_delta(damping, mass)
        else:
            delta = 0

    if omega_0 is None:
        if mass > 0:
            omega_0 = comp_omega(stiffness, mass)
        else:
            omega_0 = 0

    if omega_0 > 0:
        theta = delta/omega_0
    else:
        theta = 0

    omega_d = omega_0 * np.sqrt(1 - theta)

    # particular part
    if f_hat is None:
        x_p = np.zeros(time.size)
    else:
        x_p = f_hat * np.cos(Omega * time)

    # homogeneous part
    # check which case we need to consider
    if (omega_0 == 0 and delta == 0):
        # starrkörperbewegung
        x_h = (x_0 + (x_dot_0 / omega_0) * time) * np.exp(-delta * time)
    elif (delta == 0 and omega_0 != 0):
        # ungedämpfter fall
        x_h = x_0 * np.cos(omega_d * time) + x_dot_0 / omega_0 * np.sin(omega_d * time)
    elif 0 < delta < omega_0:
        # schwache dämpfung
        x_h = np.exp(-delta * time) * (x_0 * np.cos(omega_d * time)
                                       + x_dot_0 / omega_0 * np.sin(omega_d * time))
    elif (delta == omega_0 and delta != 0):
        # aperiodische dämpfung
        x_h = (x_0 + (x_dot_0 / omega_0) * time) * np.exp(-delta * time)
    else:
        x_h = np.zeros(time.size)

    return (x_h + x_p), x_h, x_p

@dataclass
class System:
    """ This class describes the system to solve.
    Parameters
    ----------
    `T_END` : int
        End time of the Simulation
    `USE_STATIC_POSITION` : bool
        Switch to select deformed position as initial one
    `mass`, `damping`, `stiffness` : np.zeros(2,2)
        Mass, Damping and Stiffness of the System.

    Returns
    -------
    twoMassSystem
        If values are defined properly can also be used for One Mass Systems.
    """
    T_END: int = 500  # in s
    T_SAMPLE: float = 0.05
    USE_STATIC_POSITION: bool = False

    mass = np.zeros([2,2])
    damping = np.zeros([2,2])
    stiffness = np.zeros([2,2])

#     mass = np.array([[20, 0],
#                      [0, 40]])
#    damping = np.array([[2, -1],
#                        [-1, 1]])
#    stiffness = np.array([[20, -10],
#                          [-10, 10]])


@dataclass
class Excitation:
    """ This class shall be for the excitation.
    Parameters
    ----------
    `EXCITATION` : str {'jump', 'linear', 'periodic'}
        Excitation Profile for f(t)
    `FREQUENCY` : str {'constant', 'time_dependent'}
        Frequency of Omega(t)
    `f_hat` : np.array(2,1)
        Use it for the fixed part of the excitation.
    `u_mass` : np.array(2,1)
        Describes the excitation mass on the degrees, first x_1 and then x_2.
        Only one entry should be set.
    `m` : float {0}
        Gradient of Omega
    `n` : float {0}
        Ax distance for Omega

    Returns
    -------
    excitation
        Class with the excitation settings and values.
    """
    EXCITATION: str = "jump"  # 0: jump, 1: linear, 2: periodic
    FREQUENCY: str = "constant"  # 0: constant, 1: time_dependent

    f_hat = np.zeros([2,1])
    u_mass = np.zeros([2,1])
    m: float = 0 # gradient
    n: float = 0 # constant


def external_force(t, data_system, data_excitation):
    """ Compute the external force at time `t`.
    Parameters
    ----------
    `t`
        Time array implicit given over solve_ivp
    `data_system` : `System`
    `data_excitation` : `Excitation`

    Returns
    -------
    np.array
        (`System.f_hat` + `System.u_mass` * `System.omega_ext`^2) * `f_t`

        `f_t` can be in the form of {periodic, jump, linear}
    """
    if data_excitation.FREQUENCY == "constant":
        omega_ext = data_excitation.n
    elif data_excitation.FREQUENCY == "time_dependent":
        # Omega(t) = m * t + n
        omega_ext = data_excitation.m * t + data_excitation.n
    else:
        print("Wrong input for EXCITATION, use either 'constant' or 'time_dependent'.")
        quit()

    if data_excitation.EXCITATION == "jump":
        if t < data_system.T_END/10:
            f_t = 0
        else:
            f_t = 1
    elif data_excitation.EXCITATION == "linear":
        f_t = t
    elif data_excitation.EXCITATION == "periodic":
        if data_excitation.FREQUENCY == "time_dependent":
            omega_ext_cos = 0.5 * data_excitation.m * t + data_excitation.n
            f_t = np.cos(omega_ext_cos * t)
        else:
            f_t = np.cos(omega_ext * t)
    else:
        print("Wrong input for EXCITATION, use either 'linear', 'jump' or 'periodic'.")
        quit()

    return np.array((data_excitation.f_hat + data_excitation.u_mass * omega_ext**2) * (f_t))


def state_space(t, u, data_system, data_excitation):
    """ Defines State Space Representation
    Parameters
    ----------
    `t`
        Time array implicit given over solve_ivp
    `u` : np.array(4,1)
        Initial opsition [v_2, v_1, x_2, x_1]
    `data_system` : `System`
    `data_excitation` : `Excitation`

    Returns
    -------
    np.array
        [a_2, a_1, v_2, v_1]
        State space representation.
    """
    x_dot_2, x_dot_1, x_2, x_1 = u

    F = external_force(t, data_system, data_excitation)

    m_inv_times_c = - np.linalg.lstsq(data_system.mass, data_system.stiffness, rcond=None)[0]
    m_inv_times_k = - np.linalg.lstsq(data_system.mass, data_system.damping, rcond=None)[0]
    m_inv_times_f = np.linalg.lstsq(data_system.mass, F, rcond=None)[0]

    third_state = (np.dot(m_inv_times_c[0, :], [x_1, x_2])
                   + np.dot(m_inv_times_k[0, :], [x_dot_1, x_dot_2])
                   + m_inv_times_f[0])
    fourth_state = (np.dot(m_inv_times_c[1, :], [x_1, x_2])
                    + np.dot(m_inv_times_k[1, :], [x_dot_1, x_dot_2])
                    + m_inv_times_f[1])

    return [fourth_state, third_state, x_dot_2, x_dot_1]


def plot_time_series(ivp_solution):
    """ Function for the Time Series Plot
    Parameters
    ----------
    `ivp_solution` :
        Solution of the solve_ivp function.
    """
    fig, ax = plt.subplots()
    plt.plot(ivp_solution.t, ivp_solution.y[3], label='displacement-1')
    plt.plot(ivp_solution.t, ivp_solution.y[2], label='displacement-2')
    plt.legend(loc='upper right')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.set_title("Displacement over time")
    plt.show()


def plot_stft(ivp_solution, data_system):
    """ Function for the STFT Plot
    Parameters
    ----------
    `ivp_solution` :
        Solution of the solve_ivp function
    `data_system` : `System`
        Instance of the System which was computed, for the sample time
    """
    fig, ax = plt.subplots()
    f, time, Zxx = stft(ivp_solution.y[2], 1/data_system.T_SAMPLE)
    plt.pcolormesh(time, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def plot_excitation(data_system, data_excitation):
    """ Function for the Excitation Plot
    Parameters
    ----------
    `data_system` : `System`
        Instance of the System which was computed, for the sample time
    `data_excitation` : `Excitation`
    """
    time = np.arange(0, data_system.T_END, data_system.T_SAMPLE)
    ext_force = np.zeros([len(time), 2])
    for i,val in enumerate(time):
        ext_force[i,:] = external_force(val, data_system, data_excitation)

    fig, ax = plt.subplots()
    plt.plot(time, ext_force[:,0], label='Excitation-1')
    plt.plot(time, ext_force[:,1], label='Excitation-2')
    plt.legend(loc='upper right')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Excitation")
    ax.set_title("Excitation over time")
    plt.show()
