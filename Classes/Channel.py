import numpy as np


class Channel:
    def __init__(self, freq_dev, channel_coeffs, fs, lr=0.1):
        self.freq_dev = freq_dev
        self.fs = fs
        self.lr = lr
        self.channel_coeffs = channel_coeffs

    def apply_chanel_on_sig(self, sig_to_go_through_channel):
        N = len(sig_to_go_through_channel)
        t = np.array([ind / self.fs for ind in range(N)])
        freq_deviation_samples = np.exp(1j * 2 * np.pi * t * self.freq_dev)
        sig_after_channel_coeffs = np.convolve(sig_to_go_through_channel, self.channel_coeffs, 'same')
        sig_after_channel = np.multiply(sig_after_channel_coeffs, freq_deviation_samples)
        return sig_after_channel

    def update_channel_estimation(self, theo_sig, observed_res):
        observed_res = observed_res[:len(theo_sig)]
        observed_res_fft = np.fft.fft(observed_res)
        theo_sig_fft = np.fft.fft(theo_sig)

        # Compute the frequency vector
        freq = np.fft.fftfreq(len(observed_res), 1.0 / len(observed_res))

        # Compute the phase shift due to frequency deviation
        phase_shift = np.exp(-1j * 2 * np.pi * freq * self.freq_dev)

        # Apply the phase shift to the received signal
        r_freq_shifted = np.multiply(observed_res_fft, phase_shift)

        # Compute the cross-correlation between the transmitted and received signals
        x_corr = np.fft.ifft(theo_sig_fft.conj() * r_freq_shifted)

        # Compute the channel coefficients and update coeffs
        estimated_coeffs = x_corr[:len(self.channel_coeffs)] / np.abs(theo_sig_fft[:len(self.channel_coeffs)]) ** 2
        self.channel_coeffs = self.channel_coeffs + self.lr * 2 * estimated_coeffs
        self.channel_coeffs = self.channel_coeffs / np.sqrt(np.sum(np.abs(self.channel_coeffs) ** 2))
        # Compute the frequency deviation and update freq_dev
        delta_f = -np.angle(np.mean(r_freq_shifted)) / (2 * np.pi * len(observed_res))
        self.freq_dev = self.freq_dev + self.lr * delta_f



if __name__ == "__main__":
    lengthToTake = 100000
    indToPlaceOne = 50000
    channel_coeffs = np.array([0, 1, 0, 0, 0])
    freq_dev = np.array([0.5])
    fs = 1
    channel = Channel(freq_dev, channel_coeffs, fs, 1)
    sig = np.zeros([lengthToTake])
    sig[indToPlaceOne] = 1
    sigAfterChannel = channel.apply_chanel_on_sig(sig)
    received = np.zeros([lengthToTake])
    received[indToPlaceOne + 1] = 1
    channel.update_channel_estimation(sig, received)

    print(
        f"freq_dev = {np.array2string(channel.freq_dev)} and channel_coeffs = {np.array2string(channel.channel_coeffs)}")
