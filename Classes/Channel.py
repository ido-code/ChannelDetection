import numpy as np


class Channel:
    def __init__(self, freq_dev, channel_coeffs, fs, lr_coeffs = 0.01,lr_freq = 0.1):
        self.freq_dev = freq_dev
        self.fs = fs
        self.lr_coeffs = lr_coeffs
        self.lr_freq = lr_freq
        self.channel_coeffs = channel_coeffs

    def apply_chanel_on_sig(self, sig_to_go_through_channel):
        N = len(sig_to_go_through_channel)
        t = np.array([ind / self.fs for ind in range(N)])
        freq_deviation_samples = np.exp(1j * 2 * np.pi * t * self.freq_dev)
        sig_after_channel = np.multiply(sig_to_go_through_channel, freq_deviation_samples)
        sig_after_channel_coeffs = np.convolve(sig_after_channel, self.channel_coeffs, 'same')
        return sig_after_channel_coeffs

    def update_channel_estimation(self, desired_signal, observed_signal):
        length_of_sig = len(desired_signal)
        observed_signal = observed_signal[:length_of_sig]
        # Initialize an array to store the filter output

        t = np.array([ind / self.fs for ind in range(length_of_sig)])
        freq_deviation_samples = np.exp(-1j * 2 * np.pi * t * self.freq_dev)
        observed_signal_shifted = np.multiply(observed_signal, freq_deviation_samples)
        phase_diff = np.mean(np.diff(np.unwrap(np.angle(np.multiply(observed_signal_shifted,np.conj(desired_signal))))))
        self.freq_dev += self.lr_freq*phase_diff / (2 * np.pi)

        for time_ind in range(length_of_sig - len(self.channel_coeffs)):
            filter_delay = int((len(self.channel_coeffs) - 1) / 2)
            if(time_ind < filter_delay):
                continue
            # Calculate the error between the filter output and the reference signal
            curr_samples = observed_signal_shifted[time_ind - filter_delay:time_ind+len(self.channel_coeffs) - filter_delay]
            filter_output = np.dot(np.flip(curr_samples),self.channel_coeffs)
            error = desired_signal[time_ind] - filter_output
            # Update the filter coefficients using the LMS update rule
            self.channel_coeffs += self.lr_coeffs * error * np.flip(np.conj(curr_samples))




if __name__ == "__main__":
    lengthToTake = 100000
    indToPlaceOne = 50000
    channel_coeffs = np.array([0 +1j*0, 1+1j*0, 0+1j*0])
    freq_offset = 0.2
    freq_dev = np.array([freq_offset])
    fs = 1
    lr_coeffs = 0.01
    lr_freq = 0.1
    complex_coeffs_init = np.array([1,0.2*1j,0])
    channel = Channel(np.array([0.0]), complex_coeffs_init, fs, lr_coeffs,lr_freq)
    sig = np.random.normal(size=1000) + 1j * np.random.normal(size=1000)
    sigAfterChannel = channel.apply_chanel_on_sig(sig)

    channel_output = np.convolve(sig, channel_coeffs, mode='same')
    received = channel_output * np.exp(1j * 2 * np.pi * (freq_offset / fs) * np.arange(len(channel_output)))

    padded_samples = 5
    zero_padded = np.zeros(padded_samples) + 1j*np.zeros(padded_samples)
    received = np.concatenate([received,zero_padded])
    for ind in range(200):
        channel.update_channel_estimation(sig, received)

        print(
            f"freq_dev = {np.array2string(channel.freq_dev)} and channel_coeffs = {np.array2string(channel.channel_coeffs)}")
