import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import find_peaks, windows, tf2zpk
from scipy.signal import butter, cheby1, freqz, lfilter, find_peaks, windows
import matplotlib.pyplot as plt


class SignalProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sample_rate = None
        self.signal = None
        self.trimmed_signal = None


    def load_file(self):
        """This function load the WAV file and store its sample rate and signal."""

        try:
            self.sample_rate, self.signal = wav.read(self.file_path)
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        print(f"Loaded file is: {self.file_path}")
        print(f"Sample rate is: {self.sample_rate}")
        print(f"Signal length is: {len(self.signal)}")

        self.signal = self.signal.astype(float)


    def plot_signal(self, signal, title, color='blue'):
        """Plot the given signal."""

        plt.figure(figsize=(12, 6))
        plt.plot(signal, label=title, color=color)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()


    def trim_trailing_zeros(self):
        """Trim trailing zeros from the signal."""

        non_zero_indices = np.where((self.signal > 200) | (self.signal < -200))[0]

        print(f"Non zero indice = {non_zero_indices}")

        if len(non_zero_indices) > 0:
            last_non_zero_index = non_zero_indices[-1]
            self.trimmed_signal = self.signal[:last_non_zero_index + 1]
        else:
            self.trimmed_signal = self.signal

        print(f"Trimmed signal length: {len(self.trimmed_signal)}")


    def save_trimmed_file(self, output_path):
        """Save the trimmed signal to a new WAV file."""

        if self.trimmed_signal is not None:
            wav.write(output_path, self.sample_rate, self.trimmed_signal)
            print(f"Trimmed file saved to: {output_path}")
        else:
            print("Error: Trimmed signal is empty. Perform trimming first.")


    def calculate_global_extremes(self, signal):
        """Calculate and print the global maximum and minimum of the signal."""

        max_value = np.max(signal)
        min_value = np.min(signal)

        print(f"Global Maximum: {max_value}")
        print(f"Global Minimum: {min_value}")

        return max_value, min_value


    def calculate_local_extremes(self, signal):
        """Calculate local maxima and minima of the signal."""

        # Find local maxima
        local_maxima_indices = find_peaks(signal)[0]
        local_maxima_values = signal[local_maxima_indices]

        # Find local minima
        local_minima_indices = find_peaks(-signal)[0]
        local_minima_values = signal[local_minima_indices]

        return local_maxima_values, local_maxima_indices, local_minima_values, local_minima_indices


    def analyze_extremes(self):
        """Analyze and plot global and local extremes for the trimmed signal."""

        if self.trimmed_signal is None:
            print("Error: Trimmed signal is not available. Perform trimming first.")
            return

        # Global extremes
        global_max, global_min = self.calculate_global_extremes(self.trimmed_signal)
        global_max_index = np.argmax(self.trimmed_signal)
        global_min_index = np.argmin(self.trimmed_signal)

        # Local extremes
        local_max_values, local_max_indices, local_min_values, local_min_indices = self.calculate_local_extremes(self.trimmed_signal)

        # Plot signal with marked extremes
        plt.figure(figsize=(12, 6))
        plt.plot(self.trimmed_signal, label="Trimmed Signal", color='blue')
        plt.scatter(local_max_indices, local_max_values, color='red', label="Local Maxima", zorder=5)
        plt.scatter(local_min_indices, local_min_values, color='green', label="Local Minima", zorder=5)
        plt.scatter(global_max_index, global_max, color='orange', label="Global Maximum", zorder=6, edgecolor='black')
        plt.scatter(global_min_index, global_min, color='purple', label="Global Minimum", zorder=6, edgecolor='black')
        plt.title("Signal with Extremes")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()


    def calculate_mean(self, signal):
        """Calculate the mean of the given signal."""

        if len(signal) == 0:
            print("Error: The signal is empty.")
            return None
        
        mean_value = np.mean(signal)

        print(f"The mean of the signal is: {mean_value}")

        return mean_value
    

    def calculate_median_frequency(self, signal):
        """
        Calculate the median frequency of the given signal.
        Median frequency is the frequency that divides the power spectrum into two equal halves.
        """

        if len(signal) == 0:
            print("Error: The signal is empty.")
            return None

        # Compute the FFT of the signal
        freqs = np.fft.rfftfreq(len(signal), d=1/self.sample_rate)
        fft_values = np.fft.rfft(signal)                       

        # Compute the Power Spectral Density (PSD)
        psd = np.abs(fft_values) ** 2

        # Compute cumulative sum of the PSD
        cumulative_psd = np.cumsum(psd)

        # Find the frequency where cumulative power equals 50% of total power
        total_power = cumulative_psd[-1]
        median_freq_index = np.where(cumulative_psd >= total_power / 2)[0][0]

        median_frequency = freqs[median_freq_index]
        
        return median_frequency

    def calculate_dispersion(self, signal):
        """Calculate variance, standard deviation, and range of the signal."""
        if len(signal) == 0:
            print("Error: The signal is empty.")
            return None, None, None

        # Convert signal to 64-bit float to avoid overflow issues
        signal = np.array(signal, dtype=np.float64)

        # Check for NaNs or Infinities in the signal
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print("Warning: The signal contains NaN or Inf values. These will be ignored in calculations.")
            signal = signal[np.isfinite(signal)]  # Remove NaN or Inf values

        # Calculate variance and standard deviation
        variance = np.var(signal)
        std_deviation = np.sqrt(variance)

        # Handle the range calculation more gracefully
        signal_range = np.max(signal) - np.min(signal)
        if signal_range < 0:
            print(f"Warning: The calculated range is negative ({signal_range}). This might be due to unusual signal behavior.")

        # Output the results
        print(f"Variance: {variance}")
        print(f"Standard Deviation: {std_deviation}")
        print(f"Range: {signal_range}")

        return variance, std_deviation, signal_range

    

    def plot_histogram(self, signal, bins=50, title="Signal Histogram"):
        """Plot a histogram of the signal's amplitude values."""
        plt.figure(figsize=(10, 6))
        plt.hist(signal, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(title)
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


    def calculate_zero_crossings(self, signal):
        """Calculate the number of zero crossings in the signal."""
        if len(signal) == 0:
            print("Error: The signal is empty.")
            return 0

        # Count sign changes between consecutive samples
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)

        return zero_crossings


    def plot_autocorrelation(self, signal, title="Autocorrelation of Signal"):
        """Calculate and plot the autocorrelation of the signal."""
        if len(signal) == 0:
            print("Error: The signal is empty.")
            return

        # Calculate autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        lags = np.arange(-len(signal) + 1, len(signal))

        # Plot autocorrelation
        plt.figure(figsize=(12, 6))
        plt.plot(lags, autocorr, color='purple')
        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.grid(True)
        plt.show()

    def compute_spectrum(self, signal, window_size, window_type='rectangular'):
        """ Compute the spectrum of the signal using a specified window function. """

        # Determine the window function
        if window_type == 'rectangular':
            window = np.ones(window_size)
        elif window_type == 'hamming':
            window = windows.hamming(window_size)
        elif window_type == 'hann':
            window = windows.hann(window_size)
        elif window_type == 'blackman':
            window = windows.blackman(window_size)
        elif window_type == 'flat_top':
            window = windows.flattop(window_size)
        elif window_type == 'chebyshev':
            window = windows.chebwin(window_size, at=100)  # Set attenuation to 100 dB
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

        # Divide signal into chunks
        num_chunks = len(signal)
        spectra = []

        for i in range(num_chunks):
            chunk = signal[i * window_size:(i + 1) * window_size]
            if len(chunk) < window_size:
                break
            
            # Apply window to the chunk
            windowed_chunk = chunk * window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed_chunk)
            spectra.append(np.abs(fft_result))

        # Average the spectra over all chunks
        avg_spectrum = np.mean(spectra, axis=0)
        freqs = np.fft.rfftfreq(window_size, d=1/self.sample_rate)

        return freqs, avg_spectrum

    
    def plot_spectrum(self, freqs, spectrum, window_type):
        """ Plot the spectrum for a given window type. """

        plt.figure(figsize=(12, 6))
        plt.plot(freqs, 20 * np.log10(spectrum), label=f"Window: {window_type}")
        plt.title(f"Spectrum using {window_type} window")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.legend()
        plt.show()


    def analyze_spectrum(self):
        """
        Analyze the spectrum of the trimmed signal with different window functions
        and overlay all windows in the same plot.
        """
        if self.trimmed_signal is None:
            print("Error: Trimmed signal is not available. Perform trimming first.")
            return

        # Parameters
        window_size = 1024
        window_types = ['rectangular', 'hamming', 'hann', 'blackman', 'flat_top', 'chebyshev']
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        plt.figure(figsize=(12, 6))
        
        for window_type, color in zip(window_types, colors):
            freqs, spectrum = self.compute_spectrum(self.trimmed_signal, window_size, window_type)
            plt.plot(freqs, 20 * np.log10(spectrum), label=f"Window: {window_type}", color=color)

        # Plot settings
        plt.title("Spectral Analysis with Different Windows")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.legend()
        plt.show()


    def butter_bandpass(self, lowcut, highcut, order=4):
        """Create a Butterworth bandpass filter."""
        b, a = butter(order, [lowcut, highcut], btype='band', fs=self.sample_rate)
        return b, a
    

    def chebyshev_bandpass(self, lowcut, highcut, rp=1, order=4):
            """Create a Chebyshev bandpass filter."""
            b, a = cheby1(order, rp, [lowcut, highcut], btype='band', fs=self.sample_rate)
            return b, a


    def apply_filter(self, signal, b, a):
        """Apply a bandpass filter to the signal."""
        return lfilter(b, a, signal)
    

    def plot_filter_response(self, b, a, filter_type):
        """Plot the frequency response of a filter."""
        w, h = freqz(b, a, worN=2000)
        plt.figure(figsize=(12, 6))
        plt.plot(w, abs(h), label=f"{filter_type} Bandpass Filter")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.title(f'Frequency Response of {filter_type} Filter')
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_pole_zero(self, b, a, label):
        poles = np.roots(a)
        zeros = np.roots(b)

        plt.figure(figsize=(8, 8))

        # Plot poles and zeros
        plt.scatter(np.real(zeros), np.imag(zeros), s=50, label='Zeros', color='blue', marker='o')
        plt.scatter(np.real(poles), np.imag(poles), s=50, label='Poles', color='red', marker='x')

        # Draw unit circle
        unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', label='Unit Circle')
        plt.gca().add_artist(unit_circle)

        # Axes and title
        plt.axhline(0, color='gray', lw=1)
        plt.axvline(0, color='gray', lw=1)
        plt.title(f'Pole-Zero Plot: {label}')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.axis('equal')
        plt.show()



    def analyze_filtered_spectrum(self, lowcut, highcut, order=4):
        """Analyze the spectrum of the signal after applying both Butterworth and Chebyshev filters."""
        if self.trimmed_signal is None:
            print("Error: Trimmed signal is not available. Perform trimming first.")
            return

        # Create filters for both Butterworth and Chebyshev
        b_butter, a_butter = self.butter_bandpass(lowcut, highcut, order)
        b_cheby, a_cheby = self.chebyshev_bandpass(lowcut, highcut, order)

        # Plot the frequency response of both filters on the same plot
        plt.figure(figsize=(12, 6))

        # Butterworth filter response
        w_butter, h_butter = freqz(b_butter, a_butter, worN=2000)
        plt.plot(w_butter, abs(h_butter), label='Butterworth Filter', color='blue')

        # Chebyshev filter response
        w_cheby, h_cheby = freqz(b_cheby, a_cheby, worN=2000)
        plt.plot(w_cheby, abs(h_cheby), label='Chebyshev Filter', color='orange')

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.title('Frequency Response of Butterworth and Chebyshev Filters')
        plt.grid(True)
        plt.legend()
        plt.show()

        self.plot_pole_zero(b_butter, a_butter, 'Butterworth Filter')
        self.plot_pole_zero(b_cheby, a_cheby, 'Chebyshev Filter')

        # Apply the filters to the signal
        filtered_signal_butter = self.apply_filter(self.trimmed_signal, b_butter, a_butter)
        filtered_signal_cheby = self.apply_filter(self.trimmed_signal, b_cheby, a_cheby)

        # Compute and plot the spectrum of the filtered signals
        freqs_butter, spectrum_butter = self.compute_spectrum(filtered_signal_butter, window_size=1024, window_type='hamming')
        freqs_cheby, spectrum_cheby = self.compute_spectrum(filtered_signal_cheby, window_size=1024, window_type='hamming')

        # Plot the spectrum of both filtered signals on the same plot
        plt.figure(figsize=(12, 6))
        plt.plot(freqs_butter, 20 * np.log10(spectrum_butter), label="Butterworth Filtered Signal", color='blue')
        plt.plot(freqs_cheby, 20 * np.log10(spectrum_cheby), label="Chebyshev Filtered Signal", color='orange')

        plt.title('Spectrum of Filtered Signals (Butterworth vs Chebyshev)')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.legend()
        plt.show()
        

    def process(self, output_path):
        """Run the entire process."""
        try:
            self.load_file()
            self.plot_signal(self.signal, "Original Signal")
            self.trim_trailing_zeros()
            self.plot_signal(self.trimmed_signal, "Trimmed Signal", color='orange')
            self.analyze_extremes()

            # Calculate mean of the trimmed signal
            print("\n--- Mean of the signal---")
            trimmed_mean = self.calculate_mean(self.trimmed_signal)
            print(f"Mean of Trimmed Signal: {trimmed_mean}")

            # Calculate median frequency of the trimmed signal
            print("\n--- Median Frequency of the Signal ---")
            median_frequency = self.calculate_median_frequency(self.trimmed_signal)
            if median_frequency:
                print(f"Median Frequency of Trimmed Signal: {median_frequency} Hz")

            # Calculate dispersion of the trimmed signal
            print("\n--- Trimmed Signal Dispersion ---")
            trimmed_variance, trimmed_std_dev, trimmed_range = self.calculate_dispersion(self.trimmed_signal)

            # Calculate zero crossings of the trimmed signal
            print("\n--- Zero Crossings ---")
            zero_crossings = self.calculate_zero_crossings(self.trimmed_signal)
            print(f"Number of Zero Crossings: {zero_crossings}")
            
            # Plot histogram of the trimmed signal
            self.plot_histogram(self.trimmed_signal, title="Histogram of Trimmed Signal")

            # Plot autocorrelation of the trimmed signal
            self.plot_autocorrelation(self.trimmed_signal)

            # This one is used to analyze the spectrum of the trimmed signal
            self.analyze_spectrum()

            self.analyze_filtered_spectrum(lowcut=200.0, highcut=1000.0, order=4)
            
            self.save_trimmed_file(output_path)
        
        except Exception as e:
            print(f"Error during processing: {e}")

            
if __name__ == "__main__":
    signal_file = "Wav13.wav"
    output_file = "Wav13_trimmed.wav"

    signalProcessor = SignalProcessor(signal_file)
    signalProcessor.process(output_file)
