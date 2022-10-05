from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, approx_fprime
from parameters import *
import multiprocessing as mp
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["figure.figsize"] = (8, 6)


class Dscan:
    def __init__(self, filepath, start, end):
        """Constructor of the retrieval

        Args:
            filepath (filepath): path to data of fundamental spectrum
        """
        # Get data
        data = np.loadtxt(filepath)
        self.fundamental_data = np.abs(data-background)
        self.fundamental_data /= np.max(self.fundamental_data)
        self.amplitude_lambda = np.nan_to_num(np.sqrt(self.fundamental_data))

        # Get calibration data
        self.wl_fine = self.calibration(
            pixels, offset_fine, stepsize_fine, central_wavelength_fine)
        self.wl_rough = self.calibration(
            pixels, offset_rough, stepsize_rough, central_wavelength_rough)

        # Get glass insertion array
        self.glass_insertion = self.get_glass_insertion(
            glass_steps, glass_stepsize)

        # Switch units from wavelengths to angular frequencies
        self.omega_data = (2*np.pi*c) / self.wl_rough
        self.omega_fine = (2*np.pi*c) / self.wl_fine

        delta = np.abs(self.omega_data[-1] - self.omega_data[0])/2

        start = self.omega_data[-1]+start*delta
        end = self.omega_data[0]-end*delta

        print(start*h_bar/eV, end*h_bar/eV)

        self.exclude_end = np.argwhere(self.omega_data >= start)[-1,0]
        self.exclude_start = np.argwhere(self.omega_data <= end)[0,0]

        print(self.omega_data[self.exclude_start]*h_bar/eV, self.omega_data[self.exclude_end]*h_bar/eV)

        self.omega_data = self.omega_data[self.exclude_start:self.exclude_end]
        print(self.omega_data*h_bar/eV)
        self.amplitude_lambda = self.amplitude_lambda[self.exclude_start:self.exclude_end]

        # Resample the nonlinear spaced spectrum on a linear grid
        self.amplitude_omega, self.omega_lin = self.resample_spectrum(
            self.amplitude_lambda, self.omega_data, 10)

        # Calculate corresponding nonlinear wavelength grid
        self.wl_nonlin = (2*np.pi*c) / self.omega_lin
        
        # Find start and end indices of new linear array
        self.omega_start = np.argwhere(self.omega_lin >= self.omega_fine[-1])[0, 0]
        self.omega_end = np.argwhere(self.omega_lin <= self.omega_fine[0])[-1, 0]

        # Calculate wavevector from refractive index
        n_omega = np.flip(self.refractive_index(self.wl_nonlin))
        self.k_omega = n_omega*self.omega_lin/c

        # Initialize array for the response function
        self.response = np.zeros(self.omega_lin.shape)
        
        # Get count of workers
        self.workers = mp.cpu_count()

    @staticmethod
    def calibration(pixels, offset, stepsize, wl0):
        """Calculates calibration axis for the camera spektrometer

        Args:
            pixels (int): number of pixels on the sensor
            offset (int): number of pixels from which the central wavelength deviates
            stepsize (float): resolution of the spectrometer
            wl0 (float): central wavelength

        Returns:
            np.array: linear spaced calibration array 
        """
        start = wl0 - (pixels/2 + offset) * stepsize
        end = wl0 + (pixels/2 - offset) * stepsize
        wavelengths = np.linspace(start, end, pixels)
        return wavelengths

    @staticmethod
    def get_glass_insertion(steps, stepsize, offset=0, angle=4):
        """Calculates array for every glass step inside the beam for wedges of an angel 

        Args:
            steps (int): Number of glass steps
            stepsize (float): stepsize of wedges in direction of long side of wedgew
            angle (int, optional): Angle of wedge. Defaults to 4.

        Returns:
            np.array: tiled array for fast parallel computation
        """
        angle_in_rad = np.pi*angle/180
        length_per_step = stepsize * np.tan(angle_in_rad)
        glass_array = np.linspace(-(steps+offset)*length_per_step,
                                  (steps+offset)*length_per_step, steps)
        return np.tile(glass_array, (array_length, 1)).T

    @staticmethod
    def tapering(data, fac=5):
        """tapering function, creates smooth ends to zero on both sides of the input array

        Args:
            data (np.array): Input array
            fac (int, optional): Factor of smoothing, . Defaults to 5.

        Returns:
            np.array: array of some size but tapered on both ends
        """
        n = data.shape[0]
        alpha = int(n/fac)
        left = np.cos(np.linspace(-np.pi/2, 0, alpha))**2
        right = np.cos(np.linspace(0, np.pi/2, alpha))**2
        tapering_array = np.ones(n)
        tapering_array[:alpha] = left
        tapering_array[n-alpha:] = right
        return tapering_array*data

    def resample_spectrum(self, data, x_new, tapering_factor):
        """Interpolate data and resample it on a new omega array

        Args:
            data (np.array): Array with data
            x_new (np.array): Array with non-linear x values of the data array
            tapering_factor (int): pass through of tapering factor

        Returns:
            tuple (np.array, np.array): resampled data array and new linear array starting from zero
        """
        fkt = interp1d(x_new, data, kind="cubic")

        linear_array = np.linspace(0, omega_end, array_length+1)[1:]

        # Find start and end indices of new linear array
        start = np.argwhere(linear_array >= x_new[-1])[0, 0]
        end = np.argwhere(linear_array <= x_new[0])[-1, 0]

        resampled_spectrum = np.zeros(array_length)
        if tapering_factor==0:
            resampled_spectrum[start:end] = fkt(linear_array[start:end])
        else:
            resampled_spectrum[start:end] = self.tapering(
                fkt(linear_array[start:end]), fac=tapering_factor)
            
        return resampled_spectrum, linear_array

    def resample_trace(self, trace):
        """Resample trace for all glass insertion steps

        Args:
            trace (np.array): matrix of form (m,n) with m the number of glass steps and n the number of pixels

        Returns:
            np.array: resampled trace
        """
        resampled_trace = []
        for spectrum in trace:
            resampled_spectrum, _ = self.resample_spectrum(
                spectrum, self.omega_fine, 5)
            resampled_trace.append(resampled_spectrum)
        return np.array(resampled_trace)/np.max(resampled_trace)

    @staticmethod
    def refractive_index(lamb):
        """Calculates refractive index with Sellmeier equation for fused silica

        Args:
            lamb (float): array of wavelengths in meters

        Returns:
            np.array: array with corresponding refractive indices
        """
        lamb = lamb*1e6
        n_sq = (0.6961663*lamb**2)/(lamb**2-0.06840432)+(0.4079426*lamb**2) / \
            (lamb**2-0.11624142)+(0.8974794*lamb**2)/(lamb**2-9.8961612)
        n_sq[(n_sq+1) < 0] = 0
        n = np.sqrt(n_sq+1)
        return n#np.nan_to_num(n)

    @staticmethod
    def make_periodic(data):
        """Creates array with same size but values between [-pi,pi)

        Args:
            data (np.array): array to be unwrapped

        Returns:
            np.array: unwrapped array
        """
        x_new = data.copy()
        x_new[np.where(x_new > 0)] = x_new[np.where(x_new > 0)] % np.pi
        x_new[np.where(x_new < 0)] = x_new[np.where(x_new < 0)] % -np.pi
        return x_new

    def phase(self, values):
        """Creates a phase from given values. Values will lay equally spaced on the same 
        frequencies range as the fundamental data

        Args:
            values (np.array): array with any number of entries

        Returns:
            np.array: array with interpolated phase
        """
        #values = self.make_periodic(values)
        samples = np.linspace(self.omega_data[0], self.omega_data[-1], values.size)
        phase_omega, _ = self.resample_spectrum(values, samples, 0)
        return phase_omega

    def shg_signal(self, z, phase_omega):
        """Calculates Dscan trace in parallel from a matrix of phase arrays and a matrix of glass insertions

        Args:
            z (np.array): matrix of glass insertions
            phase_omega (np.array): array with phase values 

        Returns:
            np.array: matrix of shg signals with different amount of glass
        """
        phi = np.exp(1j*(phase_omega + z * self.k_omega))
        
        E_omega_tilde = self.amplitude_omega * phi

        E_t = fft.fft(E_omega_tilde, workers=self.workers)
        shg_t = E_t*E_t
        shg_power_omega = np.abs(fft.ifft(shg_t, workers=self.workers))**2
        return shg_power_omega

    def simulate_trace(self, values):
        """Wraps simulation for optimization

        Args:
            values (np.array): array with phase values to be optimized

        Returns:
            np.array: Simulated dscan trace
        """
        phase_omega = self.phase(values)
        dscan = self.shg_signal(self.glass_insertion, phase_omega)
        return dscan/np.max(dscan)

    def transform(self, values):
        """Transform Pulse from frequency to temporal space

        Args:
            values (np.array): array with phase values

        Returns:
            tuple (np.array, np.array, np.array, np.array): power and real part of the signal, 
            with phase and without
        """
        phi = self.phase(values)

        E = self.amplitude_omega*np.exp(1j*phi)

        signal = fft.fftshift(fft.ifft(E, workers=8))
        power = np.abs(signal)**2

        signal_ftl = fft.fftshift(fft.ifft(self.amplitude_omega, workers=8))
        power_ftl = np.abs(signal_ftl)**2
        return power, power_ftl, signal, signal_ftl

    def retrieval(self, measured_trace, params, lb):
        """Starts the retrieval from measured data

        Args:
            measured_trace (np.array): Measured and resampled trace
            params (int): How many optimization parameters
            lb (float): Regularization parameter

        Returns:
            np.array: Array with the retrieved phase values
        """
        self.measured_trace = measured_trace.copy()
        values = np.random.random(params)-0.5

        def obj(x): return self.objective(
            self.simulate_trace(x), self.measured_trace, lb, x)

        #grad = lambda x: approx_fprime(x, obj, 0.01)
        #solution = minimize(obj, values, method="CG", jac=grad, tol=1e-9)
        solution = minimize(obj, values, method="Nelder-Mead", tol=1e-7)

        print(solution.success)

        self.retrieved_values = solution.x
        self.retrieved_trace = self.simulate_trace(self.retrieved_values)

        return self.retrieved_values, self.retrieved_trace

    def objective(self, retrieved, measured, lb, x):
        """Objective function to be minimized in the optimization. L2-Norm and automatic
        calculation of the response function

        Args:
            retrieved (np.array): matrix with retrieved dscan trace
            measured (np.array): matrix with measured dscan trace
            lb (float): regularization parameter
            x (np.array): optim. values for regularization

        Returns:
            float: objective value
        """
        #retrieved = ret[:,self.omega_start:self.omega_end]
        #measured = meas[:,self.omega_start:self.omega_end]
        self.response = np.sum(measured*retrieved, axis=0) / \
            np.sum(retrieved, axis=0)**2
        obj = np.sqrt(np.sum((measured - self.response * retrieved)**2))
        # np.sqrt(np.sum((measured - self.response[:,np.newaxis] * retrieved)**2)) + lb * np.sqrt(np.sum((np.gradient(x))**2))
        return obj

    def plot_result(self, measured_trace=None, retrieved_trace=None, values=None, response=None):
        """Plot result of retrieval

        Args:
            measured_trace (np.array): matrix with measured dscan trace
            retrieved_trace (np.array): matrix with retrieved dscan trace
            values (np.array): array with retrieved phase values
        """
        
        if measured_trace is None:
            measured_trace = self.measured_trace
        if retrieved_trace is None:
            retrieved_trace = self.retrieved_trace
        if values is None:
            values = self.retrieved_values
        if response is None:
            response = self.response
            
        retrieved_trace *= response
        
        phi = self.phase(values)

        start = np.argwhere(self.omega_lin >= self.omega_data[-1])[0, 0]
        end = np.argwhere(self.omega_lin <= self.omega_data[0])[-1, 0]

        start_shg = np.argwhere(self.omega_lin >= self.omega_fine[-1])[0, 0]
        end_shg = np.argwhere(self.omega_lin <= self.omega_fine[0])[-1, 0]

        #fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        fig, ax = plt.subplot_mosaic([
                                    [0, 0, 1, 1],
                                    [0, 0, 1, 1],
                                    [2, 2, 3, 3],
                                    [2, 2, 4, 4]], figsize=(14, 14), dpi=300)
                              #layout="constrained")
        fig.suptitle('Dispersion Scan Retrieval Results')
        fig.tight_layout(pad=6.0)

        #fig.subplots_adjust(top=1.0, bottom=0.9)
        
        extent_array = [self.omega_lin[start_shg:end_shg][0]*h_bar/eV,
                        self.omega_lin[start_shg:end_shg][-1]*h_bar/eV,
                        self.glass_insertion[0, 0]*1e3,
                        self.glass_insertion[-1, 0]*1e3]

        ax[0].imshow(retrieved_trace[:, start_shg:end_shg],
                        extent=extent_array,
                        aspect="auto", cmap="turbo")
        ax[0].set_xlabel("Photon Energy $[eV]$")
        ax[0].set_ylabel("Glass Insertion $z [mm]$")
        ax[0].set_title("Retrieved Trace")
        
        ax[1].imshow(measured_trace[:, start_shg:end_shg],
                        extent=extent_array,
                        aspect="auto", cmap="turbo")
        ax[1].set_xlabel("Photon Energy $[eV]$")
        ax[1].set_ylabel("Glass Insertion $z [mm]$")
        ax[1].set_title("Measured Trace")
        
        ax[2].plot(self.omega_lin[start:end]*h_bar /
                    eV, self.amplitude_omega[start:end]**2, color="gray", label="Fundamental Spectrum")
        ax2 = ax[2].twinx()
        ax2.plot(self.omega_lin[start:end]*h_bar/
                    eV, phi[start:end], color="black", linewidth=2.0, label="Spectral Phase")
        ax2.set_ylabel("Spectral Phase [rad]")
        ax2.legend(loc=0)
        ax[2].set_xlabel("Photon Energy $[eV]$")
        ax[2].set_ylabel("Intensity")
        ax[2].legend()
        
        ax[3].plot(self.omega_lin[start_shg:end_shg]*h_bar /
                      eV, response[start_shg:end_shg], color="orange", label="Response function")
        ax[3].set_xlabel("Photon Energy $[eV]$")
        ax[3].set_ylabel("Intensity")
        ax[3].legend()

        power, power_ftl, signal, signal_ftl = self.transform(values)

        n = self.omega_lin.shape[0]
        w_s = np.max(self.omega_lin)/n
        timestep = w_s/(2*np.pi)

        t_1 = np.fft.fftfreq(n, d=timestep)
        t = np.fft.fftshift(t_1)*1e15

        pm = 200
        n = signal.shape[0]//2
        
        #power /= np.trapz(power)
        #power_ftl /= np.trapz(power_ftl)

        ax[4].plot(t[n-pm:n+pm], power[n-pm:n+pm], "-", label="With Phase")
        ax[4].plot(t[n-pm:n+pm], power_ftl[n-pm:n+pm], "-", label="FTL")
        ax[4].set_xlabel("Time $t [fs]$")
        ax[4].set_ylabel("Intensity $|E(t)|^2$")
        ax[4].legend()

        plt.show()
