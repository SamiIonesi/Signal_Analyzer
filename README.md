# Signal_Analyzer
This repository is ment to describe some operations that are been done on a signal.

## The story
This project is intended to extract an audio signal from a wav file and then perform certain operations on it, such as extreme values, average, Laplace transform, signal filtering, and so on.

## Python Class

I have created a class to manage all the requirements and that is:

```py
class SignalProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sample_rate = None
        self.signal = None
        self.trimmed_signal = None
```

## Graphical representation of the signal
> Requirement: Graphically represent the signal. Extract the useful part of the signal if necessary (eliminate
the 0 values ​​at the end of the file). Create a new file that will be used further. <br>

Removing trailing zeros (or almost zeros, it's hard to get exactly 0) from an audio file (in this case, a WAV file) refers to the process of removing the portion of the signal that is completely silent (no amplitude) at the end of the recording.

For this specific requirement I have created next member functions of the class to solve the problem:

### 1. Load File

```py
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

```

### 2. Plot signal

```py
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
```

### 3. Trim trailing zeros

```py
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
```

### 4. Save trimmed file

```py
def save_trimmed_file(self, output_path):
    """Save the trimmed signal to a new WAV file."""

    if self.trimmed_signal is not None:
        wav.write(output_path, self.sample_rate, self.trimmed_signal)
        print(f"Trimmed file saved to: {output_path}")
    else:
        print("Error: Trimmed signal is empty. Perform trimming first.")
```

### 5. Final process

```py
def process(self, output_path):
    """Run the entire process: load, trim, plot, and save."""

    self.load_file()
    self.plot_signal(self.signal, "Original Signal")
    self.trim_trailing_zeros()
    self.plot_signal(self.trimmed_signal, "Trimmed Signal", color='orange')
    self.save_trimmed_file(output_path)
```


And the main function is:

```py
if __name__ == "__main__":
    signal_file = "Wav13.wav"
    output_file = "Wav13_trimmed.wav"

    signalProcessor = SignalProcessor(signal_file)
    signalProcessor.process(output_file)
```

## Signal operations

> Requirement: Determine the extreme values, mean, median, dispersion and represent the histogram.

### Extreme values
**The extreme values** ​​of a signal refer to the points where the signal reaches its maximum or minimum amplitude. These are important features in signal analysis because they provide information about the intensity, dynamics, and nature of the signal. <br>

#### Example
This is an example from a portion of a signal in which we can see local maxima, local minima, global maximum and global minimum. <br>
![Extrems_of_a_signal](https://github.com/user-attachments/assets/5cc70f97-a7b5-4fbd-b127-3eb964b38a35)

#### Extreme values Python functions
For this requirement I have created some functions and they are:

##### - Global extreme values
```Python
def calculate_global_extremes(self, signal):
    """Calculate and print the global maximum and minimum of the signal."""

    max_value = np.max(signal)
    min_value = np.min(signal)

    print(f"Global Maximum: {max_value}")
    print(f"Global Minimum: {min_value}")
    
    return max_value, min_value
```

##### - Local extreme values
```Python
def calculate_local_extremes(self, signal):
    """Calculate local maxima and minima of the signal."""

    # Find local maxima
    local_maxima_indices = find_peaks(signal)[0]
    local_maxima_values = signal[local_maxima_indices]

    # Find local minima
    local_minima_indices = find_peaks(-signal)[0]
    local_minima_values = signal[local_minima_indices]

    return local_maxima_values, local_maxima_indices, local_minima_values, local_minima_indices
```

##### - Plot extremes
```Python
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
```

### Mean of the signal
**The mean** of a signal is the arithmetic average of all its amplitude values. It provides a measure of the signal's central tendency or the "average level" of the signal's amplitude.

#### Mean formula
The mathematical relation for the mean of a function f(t) is defined as: <br>
![Mean_of_Signal](https://github.com/user-attachments/assets/26912a65-d4e9-4554-b99c-ef47be35d67a)

#### Mean Python function
The function that I created for this requirement is:

```Python
def calculate_mean(self, signal):
    """Calculate the mean of the given signal."""

    if len(signal) == 0:
        print("Error: The signal is empty.")
        return None
    
    mean_value = np.mean(signal)

    print(f"The mean of the signal is: {mean_value}")

    return mean_value
```



### Median frequency of the signal
The median frequency is a measure used in signal analysis to determine the frequency at which the power spectrum of a signal is divided into two equal halves. It is a key parameter for understanding the distribution of energy across the frequency spectrum.

#### How it works

##### 1. Frequency Spectrum
The signal is first converted to the frequency domain using the Fast Fourier Transform (FFT).

##### 2. Power Spectral Density (PSD)
The squared magnitude of the FFT values gives the power at each frequency.

##### 3. Cumulative Power
The cumulative sum of the PSD is computed.

##### 4. Median Frequency
The frequency where the cumulative power reaches 50% of the total power is identified as the median frequency.

#### Median frequency Python function
For this requirement I have createt next member function of the class:

```Python
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
```

### Dispersion of a signal
The dispersion of a signal quantifies how its values are spread out or scattered around a central value (usually the mean). Dispersion provides insight into the variability of the signal. Key statistical measures of dispersion include variance, standard deviation, and range.

## Zero crossings
Zero crossings refer to the points where a signal transitions through the value of zero on its amplitude axis. Specifically, a zero crossing occurs when a signal changes sign from positive to negative or vice versa. This concept is widely used in signal processing and related fields to analyze and characterize signals. <br>

[![A zero-crossing in a line graph of a waveform representing voltage over time](https://github.com/user-attachments/assets/fbbd5567-c240-4810-b887-1c7442044ccc)](https://en.wikipedia.org/wiki/Zero_crossing)

## Autocorrelation
The autocorrelation of a signal measures the similarity of the signal with a delayed version of itself over varying time delays. It quantifies how a signal correlates with itself as the time shift (or lag) increases, providing insight into the signal's periodicity, structure, and temporal relationships.
