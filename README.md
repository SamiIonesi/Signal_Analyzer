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

## 1. Graphical representation of the signal
> Requirement: Graphically represent the signal. Extract the useful part of the signal if necessary (eliminate
the 0 values ​​at the end of the file). Create a new file that will be used further. <br>

Removing trailing zeros (or almost zeros, it's hard to get exactly 0) from an audio file (in this case, a WAV file) refers to the process of removing the portion of the signal that is completely silent (no amplitude) at the end of the recording.

For this specific requirement I have created next member functions of the class to solve the problem:

### 1.1 Signal graphical representation Python functions

#### 1.1.1 Load File

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

#### 1.1.2 Plot signal

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

#### 1.1.3 Trim trailing zeros

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

#### 1.1.4 Save trimmed file

```py
def save_trimmed_file(self, output_path):
    """Save the trimmed signal to a new WAV file."""

    if self.trimmed_signal is not None:
        wav.write(output_path, self.sample_rate, self.trimmed_signal)
        print(f"Trimmed file saved to: {output_path}")
    else:
        print("Error: Trimmed signal is empty. Perform trimming first.")
```

## 2. Signal operations

> Requirement: Determine the extreme values, mean, median, dispersion and represent the histogram.

### 2.1 Extreme values
**The extreme values** ​​of a signal refer to the points where the signal reaches its maximum or minimum amplitude. These are important features in signal analysis because they provide information about the intensity, dynamics, and nature of the signal. <br>

#### 2.1.1 Example
This is an example from a portion of a signal in which we can see local maxima, local minima, global maximum and global minimum. <br>
![Extrems_of_a_signal](https://github.com/user-attachments/assets/5cc70f97-a7b5-4fbd-b127-3eb964b38a35)

#### 2.1.2 Extreme values Python functions
For this requirement I have created some functions and they are:

##### 2.1.2.1 Global extreme values
```Python
def calculate_global_extremes(self, signal):
    """Calculate and print the global maximum and minimum of the signal."""

    max_value = np.max(signal)
    min_value = np.min(signal)

    print(f"Global Maximum: {max_value}")
    print(f"Global Minimum: {min_value}")
    
    return max_value, min_value
```

##### 2.1.2.2 Local extreme values
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

##### 2.1.2.3 Plot extremes
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

### 2.2 Mean of the signal
**The mean** of a signal is the arithmetic average of all its amplitude values. It provides a measure of the signal's central tendency or the "average level" of the signal's amplitude.

#### 2.2.1 Mean formula
The mathematical relation for the mean of a function f(t) is defined as: <br>
![Mean_of_Signal](https://github.com/user-attachments/assets/26912a65-d4e9-4554-b99c-ef47be35d67a)

#### 2.2.2 Mean Python function
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

### 2.3 Median frequency of the signal
The median frequency is a measure used in signal analysis to determine the frequency at which the power spectrum of a signal is divided into two equal halves. It is a key parameter for understanding the distribution of energy across the frequency spectrum.

#### 2.3.1 How it works

##### 2.3.1.1 Frequency Spectrum
The signal is first converted to the frequency domain using the Fast Fourier Transform (FFT).

##### 2.3.1.2  Power Spectral Density (PSD)
The squared magnitude of the FFT values gives the power at each frequency.

##### 2.3.1.3 Cumulative Power
The cumulative sum of the PSD is computed.

##### 2.3.1.4 Median Frequency
The frequency where the cumulative power reaches 50% of the total power is identified as the median frequency.

#### 2.3.2 Median frequency Python function
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

### 2.4 Dispersion of a signal
The dispersion of a signal quantifies how its values are spread out or scattered around a central value (usually the mean). Dispersion provides insight into the variability of the signal. Key statistical measures of dispersion include variance, standard deviation, and range.

## 3.1 Zero crossings
Zero crossings refer to the points where a signal transitions through the value of zero on its amplitude axis. Specifically, a zero crossing occurs when a signal changes sign from positive to negative or vice versa. This concept is widely used in signal processing and related fields to analyze and characterize signals. <br>

> Requirement: Calculate the number of zero crossings.

[![A zero-crossing in a line graph of a waveform representing voltage over time](https://github.com/user-attachments/assets/fbbd5567-c240-4810-b887-1c7442044ccc)](https://en.wikipedia.org/wiki/Zero_crossing)

### 3.1.1 Zero crossings Python function

For this requirement I have created next member class function:
```Python
def calculate_zero_crossings(self, signal):
    """Calculate the number of zero crossings in the signal."""
    if len(signal) == 0:
        print("Error: The signal is empty.")
        return 0

    # Count sign changes between consecutive samples
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)

    return zero_crossings
```

## 3.2 Autocorrelation
The autocorrelation of a signal measures the similarity of the signal with a delayed version of itself over varying time delays. It quantifies how a signal correlates with itself as the time shift (or lag) increases, providing insight into the signal's periodicity, structure, and temporal relationships.

> Requirement: Plot the autocorrelation.


![Signal_autocorrelation](https://github.com/user-attachments/assets/b7983428-2ed2-4de7-9564-8f5fd45a9eee)

### 3.2.1 Autocorelation Python function

```Python
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
```

## 4. Analyze signal spectrum

The spectrum of a signal represents the distribution of its frequency components, showing how much of the signal's energy or amplitude is present at each frequency. It is a way to analyze and visualize the signal in the frequency domain, rather than the time domain.

> Requirement: Determine the signal spectrum as follows: <br>
> a) Apply the Fourier Transform on a series of samples (1 second or 1024/2048 samples). <br>
> b) Apply various time windows and repeat 4.a. for each type of window. <br>
> c) Draw conclusions regarding the signal spectrum. <br>

### 4.1 Key Aspects of a Signal's Spectrum

#### 4.1.1 Frequency Components

- Any signal can be broken down into a sum of sinusoidal components of different frequencies using mathematical tools like Fourier Transform.
- The spectrum reveals these frequencies, allowing us to see which ones contribute to the signal.

#### 4.1.2 Amplitude or Power

- The spectrum shows the amplitude (or power) of each frequency component.
- For example, a high amplitude at a specific frequency indicates that the signal has a strong contribution from that frequency.

#### 4.1.3 Bandwidth

- The range of frequencies over which the signal has significant energy is called the bandwidth.
- Narrowband signals are concentrated around a small frequency range, while wideband signals span a larger range.

### 4.2 How is the Spectrum Obtained?

The spectrum is typically computed using the Fourier Transform (FT), which transforms a signal from the time domain (variations over time) to the frequency domain (variations over frequency). Types of Fourier analysis include:

#### 4.2.1 Continuous Fourier Transform (CFT)

Used for continuous signals to get a continuous spectrum.

#### 4.2.2 Discrete Fourier Transform (DFT)

Used for discrete signals (e.g., sampled signals) to get a sampled version of the spectrum.

#### 4.2.3 Fast Fourier Transform (FFT)

An efficient algorithm to compute the DFT.

### 4.3 Types of Signals and Their Spectra

#### 4.3.1 Periodic Signals

Have a discrete spectrum consisting of harmonics (integer multiples of the fundamental frequency).
Example: A pure sine wave has a single frequency in its spectrum.

#### 4.3.2 Aperiodic Signals

Have a continuous spectrum, covering a range of frequencies.
Example: A rectangular pulse has a spectrum that spreads across multiple frequencies.

#### 4.3.3 Noisy Signals

Their spectrum often spans a wide frequency range (e.g., white noise has a uniform spectrum across all frequencies).

### 4.4 Signal Spectrum Python functions

For this requirement I have created next functions that are member functions of the class:

#### 4.4.1 Compute spectrum

```Python
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
```

#### 4.4.2 Plot spectrum

```Python
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
```

#### 4.4.3 Analyze spectrum

```Python
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
```

### 4.5 Characteristics of Each Window

| **Window Type**  | **Main Lobe Width** | **Side Lobe Attenuation** | **Use Case**             | **Impact on Spectrum**                                   |
|-------------------|---------------------|----------------------------|--------------------------|---------------------------------------------------------|
| **Rectangular**   | Narrowest          | Poor (-13 dB)              | Basic FFT               | High spectral leakage; poor resolution.                |
| **Hamming**       | Medium             | Moderate (-43 dB)          | General Use             | Good balance of leakage and resolution.                |
| **Hann**          | Wider              | Better (-44 dB)            | Harmonic Analysis       | Reduces leakage more than Hamming but at the cost of wider lobes. |
| **Blackman**      | Widest             | High (-74 dB)              | Low Signal Strength     | Excellent leakage reduction but lowest resolution.      |
| **Flat Top**      | Very Wide          | Very High (-93 dB)         | Amplitude Measurement   | Flattens peaks, accurate amplitude but low frequency resolution. |
| **Chebyshev**     | Adjustable         | Customizable (-100 dB)     | Advanced Applications   | Offers customizable trade-offs between resolution and attenuation. |

### 4.6 Conclusions

#### 4.6.1 Spectral Leakage

- Rectangular windows suffer from the highest spectral leakage, making them unsuitable for signals with close spectral components.
- Chebyshev and Flat Top windows excel at minimizing leakage but at the cost of broadening the main lobe.

#### 4.6.2 Resolution vs. Attenuation

- Blackman and Hann windows provide a good compromise, with sufficient attenuation for most practical applications.

#### 4.6.3 Amplitude Accuracy

- The Flat Top window is ideal for amplitude measurements, though its frequency resolution is poor.

#### 4.6.4 Customization

- Chebyshev windows allow fine-tuning for specific needs, such as higher attenuation or narrower main lobes.

## 5. Filters
