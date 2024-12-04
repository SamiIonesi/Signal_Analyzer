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

In signal processing, filters are systems or algorithms that modify or manipulate a signal by selectively allowing certain frequencies or components to pass through while attenuating or blocking others. Filters are used to enhance, extract, or suppress specific aspects of a signal. The primary purpose of a filter is to remove unwanted noise or interference, smooth signals, or emphasize certain frequencies or features of a signal.

> Requirement: Considering the frequency corresponding to the spectral line of maximum amplitude and another spectral line of smaller amplitude in the vicinity of the first, a numerical band-pass filter is designed to isolate the two spectral lines. <br>
> a) Determination of the transfer functions for two types of filters (of your choice). <br>
> b) Representation of the frequency characteristic or location of the poles and zeros for the two filters. <br>
> c) Representation of the spectrum of the filtered signal. <br>

Filters can be classified in various ways, but they are generally categorized by their frequency response and structure.

### 5.1 Types of Filters by Frequency Response

#### 5.1.1 Low-pass Filter (LPF)
Allows frequencies below a certain cutoff frequency to pass and attenuates higher frequencies. Used to remove high-frequency noise or smooth signals.

#### 5.1.2 High-pass Filter (HPF)
Allows frequencies above a certain cutoff frequency to pass and attenuates lower frequencies. Useful for removing low-frequency drift or noise.

#### 5.1.3 Band-pass Filter (BPF)
Allows frequencies within a specific range (band) to pass and attenuates frequencies outside that range. It is used to isolate signals within a specific frequency range.

#### 5.1.4 Band-stop Filter (BSF) or Notch Filter
Attenuates frequencies within a specific range and allows frequencies outside of that range to pass. This type of filter is used to reject unwanted frequencies, such as a particular interference frequency.

### 5.2 Types of Filters by Time Domain Characteristics

#### 5.2.1 FIR (Finite Impulse Response) Filters
The impulse response is finite, meaning it has a limited duration. FIR filters are typically stable and easy to design. They are non-recursive, meaning the output depends only on the current and past inputs.

#### 5.2.2 IIR (Infinite Impulse Response) Filters
The impulse response is infinite, meaning it theoretically continues indefinitely. IIR filters are recursive, meaning the output depends on both the current and past inputs and past outputs. They are more computationally efficient than FIR filters for certain applications but can be unstable if not designed properly.

### 5.3 Filter Characteristics

#### 5.3.1 Linear Filters
A filter is linear if it satisfies two properties: homogeneity (scaling of input results in scaling of output) and additivity (the sum of outputs equals the output of the sum of inputs). Most basic filters are linear.

#### 5.3.2 Non-linear Filters
These do not satisfy linearity properties. They are used in applications like signal restoration or image processing, where the input-output relationship is more complex. An example is the median filter used for noise reduction in images.

### 5.4 Design Approaches

#### 5.4.1 Analog Filters
These filters are implemented using analog components like resistors, capacitors, and inductors. Examples include the RC (Resistor-Capacitor) and RL (Resistor-Inductor) filters.

#### 5.4.2 Digital Filters
These filters are implemented using algorithms in digital form, typically in discrete-time systems, such as FIR and IIR filters. They are used in applications where signals are sampled and processed digitally.

### 5.5 Common Filter Applications

#### 5.5.1 Noise Reduction
Filtering out unwanted noise from a signal.

#### 5.5.2 Signal Smoothing
Removing high-frequency fluctuations (e.g., in a time-series signal).

#### 5.5.3 Feature Extraction
Emphasizing or isolating particular frequency components of a signal.

#### 5.5.4 Data Compression
Reducing unnecessary data by filtering out irrelevant parts of a signal.

#### 5.5.5 Communication Systems
Shaping signals for optimal transmission or preventing interference.

### 5.6 Transfer functions for two types of filters
To solve this requirement I have chosen two types of filters

#### 5.6.1 Filters type

##### 5.6.1.1 Butterworth filter
A Butterworth filter is a type of electronic or digital filter that is designed to have a very smooth frequency response in the passband, with no ripples. It is also known as a maximally flat filter because it provides the flattest possible amplitude response in the passband among all filter designs. This feature makes it ideal for applications where signal distortion in the passband must be minimized.

##### 5.6.1.2 Chebyshev filter
A Chebyshev filter is a type of electronic or digital filter designed to achieve a specific set of characteristics in signal processing. It is widely used because of its ability to provide a sharper transition between the passband and the stopband compared to other filters, such as Butterworth filters, for a given filter order. However, this sharpness comes at the cost of ripples in either the passband or the stopband, depending on the type of Chebyshev filter.

#### 5.6.2 Filters Python functions

##### 5.6.2.1 Butterworth filter Python function
```Python
def butter_bandpass(self, lowcut, highcut, order=4):
    """Create a Butterworth bandpass filter."""
    b, a = butter(order, [lowcut, highcut], btype='band', fs=self.sample_rate)
    return b, a
```

##### 5.6.2.2 Chebyshev filter Python function
```Python
def chebyshev_bandpass(self, lowcut, highcut, rp=1, order=4):
        """Create a Chebyshev bandpass filter."""
        b, a = cheby1(order, rp, [lowcut, highcut], btype='band', fs=self.sample_rate)
        return b, a
```

##### 5.6.2.3 Applay filter Python function
```Python
def apply_filter(self, signal, b, a):
    """Apply a bandpass filter to the signal."""
    return lfilter(b, a, signal)
```

### 5.7 Frequency characteristic
The representation of the frequency characteristic of a signal refers to the description or analysis of how the signal's energy or amplitude is distributed across different frequencies. This is a fundamental concept in signal processing, as it provides insight into the underlying structure and content of the signal in the frequency domain.

![Frequency_Characteristic](https://github.com/user-attachments/assets/3e871d3c-2e3b-407e-bc33-ab09a0819e78)

#### 5.7.1 Frequency characteristic Python function

```Python
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
```

### 5.8 Poles and zeros
Poles and zeros are fundamental concepts in the study of systems and signal processing, especially in the context of filter design, control systems, and the analysis of linear time-invariant (LTI) systems. They are specific values of a system's transfer function that provide important insights into its behavior.

#### 5.8.1 Poles and zeros Python function

```Python
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
```

#### 5.8.2 Poles and zeros Butterworth filter

![Poles Zeros_Butter](https://github.com/user-attachments/assets/48dc2296-d256-4f44-b10e-4758251fc31d)

#### 5.8.3 Poles and zeros Chebyshev filter

![Poles Zeros_Chebisev](https://github.com/user-attachments/assets/1659c1c4-3e35-47bc-b386-422a466c2a54)

### 5.9 Signal spectrum

The spectrum of a signal represents the distribution of the signal's energy or power across different frequencies. It provides a way to analyze the frequency content of the signal, which is critical for understanding its behavior, designing filters, or optimizing communication systems.

![Spectrum](https://github.com/user-attachments/assets/353714fc-4bc8-403c-9f01-a4265b713f15)

#### 5.9.1 Signal spectrum Python function

```Python
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
```



