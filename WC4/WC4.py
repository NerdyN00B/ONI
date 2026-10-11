import numpy as np
import matplotlib.pyplot as plt

def k_z1(theta, k0=1):
    return k0 * np.cos(theta)

def k_z2(theta, n, k0=1):
    return k0 * np.emath.sqrt(n**2 - (1 / n) * np.sin(theta)**2)

def rs(k_z1, k_z2):
    return (k_z1 - k_z2) / (k_z1 + k_z2)

def ts(k_z1, k_z2):
    return (2 * k_z1) / (k_z1 + k_z2)

def rp(k_z1, k_z2, n):
    return (n**2 * k_z1 - k_z2) / (n**2 * k_z1 + k_z2)

def tp(k_z1, k_z2, n):
    return (2 * n**2 * k_z1) / (n**2 * k_z1 + k_z2)

def T_s(k_z1, k_z2, n):
    return (np.real(k_z2) / np.real(k_z1)) * (np.abs(ts(k_z1, k_z2))**2)

def T_p(k_z1, k_z2, n):
    return n**(-2) * (np.real(k_z2) / np.real(k_z1)) * (np.abs(tp(k_z1, k_z2, n))**2)

def R_s(k_z1, k_z2):
    return np.abs(rs(k_z1, k_z2))**2

def R_p(k_z1, k_z2, n):
    return np.abs(rp(k_z1, k_z2, n))**2

# n = 0.9  # abc
n = (1 - 3.5e-5) + 1j * 1.1e-5  # def

theta = np.linspace(0, np.pi/2, 9000)

k1 = k_z1(theta)
k2 = k_z2(theta, n)

r_s = rs(k1, k2)
t_s = ts(k1, k2)
r_p = rp(k1, k2, n)
t_p = tp(k1, k2, n)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

ax1.plot(theta * 180 / np.pi, t_s, label='t_s', color='blue', 
         linestyle='dashed')
ax1.plot(theta * 180 / np.pi, t_p, label='t_p', color='red', 
         linestyle='dashed')

ax1.plot(theta * 180 / np.pi, r_s, label='r_s', color='blue')
ax1.plot(theta * 180 / np.pi, r_p, label='r_p', color='red')

ax1.legend()
ax1.set_xlabel('Angle of Incidence (degrees)')
ax1.set_xlim(0, 90)

Rs = R_s(k1, k2)
Rp = R_p(k1, k2, n)
Ts = T_s(k1, k2, n)
Tp = T_p(k1, k2, n)

ax2.plot(theta * 180 / np.pi, Ts, label='T_s', color='blue',
         linestyle='dashed')
ax2.plot(theta * 180 / np.pi, Tp, label='T_p', color='red',
         linestyle='dashed')

ax2.plot(theta * 180 / np.pi, Rs, label='R_s', color='blue')
ax2.plot(theta * 180 / np.pi, Rp, label='R_p', color='red')

ax2.legend()
ax2.set_xlabel('Angle of Incidence (degrees)')
ax2.set_xlim(0, 90)

if np.imag(n) > 0:
    ax1.semilogy()
    ax2.semilogy()
    min_reflect = np.argmin(Rp)
    angle = theta[min_reflect] * 180 / np.pi
    ax2.annotate(f"{angle:.2f}$^\circ$",
                 xy=(angle, Rp[min_reflect]), 
                 xytext=(angle, Rp[min_reflect] / 5)
                 )

plt.show()
fig.savefig('WC4_def.pdf', dpi=300)