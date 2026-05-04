import numpy as np
import matplotlib.pyplot as plt


def V_box(x, L):
   potential = np.zeros_like(x)
   potential[np.where((x < -L/2) | (x > L/2))] = 10
   return potential


def V_qho(x, k):
   return 0.5 * k * x**2


def V_ring(theta):
   return np.zeros_like(theta)


def V_free(x):
   return np.zeros_like(x)


def V_double_well(x):
   a = 0.05
   b = 2.0
   return a * x**4 - b * x**2


def V_barrier(x, V0, width):
   potential = np.zeros_like(x)
   potential[np.where(np.abs(x) < width / 2)] = V0
   return potential






# Barrier
V0_barrier_ind = 5
barrier_width_ind = 2
x_barrier = np.linspace(-5, 5, 400)
potential_barrier = V_barrier(x_barrier, V0_barrier_ind, barrier_width_ind)


x_common = np.linspace(-5, 5, 500)


L_box_combined = 4
potential_box_combined = V_box(x_common, L_box_combined)


k_qho_combined = 0.5
potential_qho_combined = V_qho(x_common, k_qho_combined)


V0_ring_combined = 1
potential_ring_combined = np.full_like(x_common, V0_ring_combined)


potential_free_combined = V_free(x_common)




V0_barrier_combined = 5
barrier_width_combined = 2
potential_barrier_combined = V_barrier(x_common, V0_barrier_combined, barrier_width_combined)


plt.figure(figsize=(12, 7))


plt.plot(x_common, potential_box_combined, linestyle='-', linewidth=2, label='Particle in a Box')
plt.vlines([-L_box_combined/2, L_box_combined/2], 0, 10, color='blue', linestyle='--', linewidth=1)


plt.plot(x_common, potential_qho_combined, linestyle='-', linewidth=2, label='Quantum Harmonic Oscillator')


plt.plot(x_common, potential_ring_combined, linestyle='-', linewidth=2, label='Particle on a Ring (Constant Potential)')


plt.plot(x_common, potential_free_combined, linestyle='-', linewidth=2, label='Free Particle')


plt.plot(x_common, potential_barrier_combined, linestyle='-', linewidth=2, label='Potential Barrier')


plt.title('Some Quantum Mechanical Potentials')
plt.xlabel('Position (x)')
plt.ylabel('Potential Energy (V(x))')
plt.ylim(-3, 11)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Combined_Potentials.png")
plt.show()

