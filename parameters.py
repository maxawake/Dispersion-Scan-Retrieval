import numpy as np

# electron volt
eV = 1.602176634e-19 # J
# reduced planck constant
h_bar = 1.054571817e-34   # J*s
# pulse duration [s] FWHM -- what I should get after the FT
T0  = 10e-15
# speed of light 
c = 299792458 # [m/s]
# bandwidth [rad/s] --> from FTL
delta_w = 2*np.pi*0.441/T0

background = 350

pixels = 2560 # [px]
offset_fine = -92 # [px]
stepsize_fine = 0.058e-9 #[m/px]
central_wavelength_fine = 400e-9 #[m]

offset_rough = 33 # [px]
stepsize_rough = 0.278e-9 #[m/px]
central_wavelength_rough = 750e-9 #[m]

glass_stepsize = 0.51e-3/2 # [m]
glass_steps = 141 # [steps]

omega_end = delta_w*128
array_length = 2**12

