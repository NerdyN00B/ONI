import numpy as np
import matplotlib.pyplot as plt
import os

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

def Ts(k_z1, k_z2, n):
    return (np.real(k_z2) / np.real(k_z1)) * (np.abs(ts(k_z1, k_z2))**2)

def Tp(k_z1, k_z2, n):
    return n**(-2) * (np.real(k_z2) / np.real(k_z1)) * (np.abs(tp(k_z1, k_z2, n))**2)

def Rs(k_z1, k_z2):
    return np.abs(rs(k_z1, k_z2))**2

def Rp(k_z1, k_z2, n):
    return np.abs(rp(k_z1, k_z2, n))**2

def FOM(theta, n):
    k1 = k_z1(theta)
    k2 = k_z2(theta, n)

    Rs_val = Rs(k1, k2)
    Rp_val = Rp(k1, k2, n)
    mean_R = (Rs_val + Rp_val) / 2
    return mean_R * np.cos(theta)

def get_n(filename):
    energy, delta, beta = np.genfromtxt(
        filename,
        delimiter='  ',
        skip_header=2,
        unpack=True
    )
    
    n = 1 - delta + 1j * beta
    return energy, n

for filename in [f for f in os.listdir('materials') if os.path.isfile(os.path.join('materials', f))]:
    print(f"Processing {filename}")
    
    energy, n = get_n(f"materials/{filename}")
    theta = np.linspace(np.pi / 2 - np.deg2rad(5), np.pi / 2, 1000) # From 85 degrees to 90 degrees in radians
    theta_grid, n_grid = np.meshgrid(theta, n)
    
    FOM_values = FOM(theta_grid, n_grid)
    Rs_values = Rs(k_z1(theta_grid), k_z2(theta_grid, n_grid))
    
    fig_fom, ax_fom = plt.subplots(dpi=300)
    fig_Rs, ax_Rs = plt.subplots(dpi=300)
    angle = np.rad2deg(theta)
    angle = 90 - angle  # Convert to angle from normal
    c_fom = ax_fom.pcolormesh(
        angle,
        energy,
        FOM_values,
        shading='auto',
        cmap='viridis',
        vmin=0,
        vmax=np.max(FOM_values) # Can be set as a fixed value so the colorbar is consistent across different materials
    )
    
    c_Rs = ax_Rs.pcolormesh(
        angle,
        energy,
        Rs_values,
        shading='auto',
        cmap='viridis'
    )
    
    fig_fom.colorbar(c_fom, ax=ax_fom, label='FOM')
    ax_fom.set_xlabel('Theta (degrees)')
    ax_fom.set_ylabel('Energy (eV)')
    ax_fom.set_title(f'FOM for {filename.strip(".csv")}')    
    fig_fom.savefig(f"FOM_{filename.strip('.csv')}.png")
    
    fig_Rs.colorbar(c_Rs, ax=ax_Rs, label='Rs')
    ax_Rs.set_xlabel('Theta (degrees)')
    ax_Rs.set_ylabel('Energy (eV)')
    ax_Rs.set_title(f'Rs for {filename.strip(".csv")}')
    fig_Rs.savefig(f"Rs_{filename.strip('.csv')}.png")