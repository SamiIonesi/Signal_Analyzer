import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import find_peaks

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
    

    def calculate_median(self, signal):
        """Calculate the median of a given signal."""

        if len(signal) == 0:
            print("Error: The signal is empty.")
            return None
        
        median_value = np.median(signal)

        print(f"The median of the signal is: {median_value}")

        return median_value
    

    def calculate_dispersion(self, signal):
        """Calculate variance, standard deviation, and range of the signal."""
        if len(signal) == 0:
            print("Error: The signal is empty.")
            return None
        
        variance = np.var(signal)
        std_deviation = np.sqrt(variance)
        signal_range = np.max(signal) - np.min(signal)

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

        print(f"Number of Zero Crossings: {zero_crossings}")
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


    def process(self, output_path):
        """Run the entire process."""

        self.load_file()
        self.plot_signal(self.signal, "Original Signal")
        self.trim_trailing_zeros()
        self.plot_signal(self.trimmed_signal, "Trimmed Signal", color='orange')
        self.analyze_extremes()

        # Calculate mean of the trimmed signal
        print("\n--- Mean of the signal---")
        trimmed_mean = self.calculate_mean(self.trimmed_signal)
        print(f"Mean of Trimmed Signal: {trimmed_mean}")

        # Calculate median of the trimmed signal
        print("\n--- Median of the signal ---")
        trimmed_median = self.calculate_median(self.trimmed_signal)
        print(f"Median of Trimmed Signal: {trimmed_median}")

        # Calculate dispersion of the trimmed signal
        print("\n--- Trimmed Signal Dispersion ---")
        trimmed_variance, trimmed_std_dev, trimmed_range = self.calculate_dispersion(self.trimmed_signal)

        # Calculate zero crossings of the trimmed signal
        print("\n--- Zero Crossings ---")
        zero_crossings = self.calculate_zero_crossings(self.trimmed_signal)
        
        # Plot histogram of the trimmed signal
        self.plot_histogram(self.trimmed_signal, title="Histogram of Trimmed Signal")

        # Plot autocorrelation of the trimmed signal
        self.plot_autocorrelation(self.trimmed_signal)
        
        self.save_trimmed_file(output_path)

if __name__ == "__main__":
    signal_file = "Wav13.wav"
    output_file = "Wav13_trimmed.wav"

    signalProcessor = SignalProcessor(signal_file)
    signalProcessor.process(output_file)