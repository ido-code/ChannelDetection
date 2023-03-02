import numpy as np
class Channel:
    def __init__(self,freq_dev,channel_coeffs,fs,lrForUpdate = 0.1):
        self.freq_dev = freq_dev
        self.fs = fs
        self.lr = lrForUpdate
        self.channel_coeffs = channel_coeffs
    def ApplyChanelOnSig(self,sigToGoThroughChannel):
        N = len(sigToGoThroughChannel)
        t = np.array([ind/ self.fs for ind in range(N)])
        freq_deviation_samples = np.exp(1j* 2* np.pi * t * self.freq_dev)
        SigAfterChannelCoeffs = np.convolve(sigToGoThroughChannel,self.channel_coeffs,'same')
        SigAfterChannel = np.multiply(SigAfterChannelCoeffs,freq_deviation_samples)
        return SigAfterChannel
    def Update_Channel_estimation(self,theoSig,observedRes):
        observedRes = observedRes[:len(theoSig)]
        observedResFft = np.fft.fft(observedRes)
        theoSigFft = np.fft.fft(theoSig)

        # Compute the frequency vector
        freq = np.fft.fftfreq(len(observedRes), 1.0 / len(observedRes))

        # Compute the phase shift due to frequency deviation
        phase_shift = np.exp(-1j * 2 * np.pi * freq * self.freq_dev)

        # Apply the phase shift to the received signal
        R_freq_shifted = np.multiply(observedResFft, phase_shift)

        # Compute the cross-correlation between the transmitted and received signals
        xcorr = np.fft.ifft(theoSigFft.conj() * R_freq_shifted)

        # Compute the channel coefficients and update coeffs
        H = xcorr[:len(self.channel_coeffs)] / np.abs(theoSigFft[:len(self.channel_coeffs)]) ** 2
        self.channel_coeffs = self.channel_coeffs + self.lr * 2 * H
        self.channel_coeffs = self.channel_coeffs / np.sqrt(np.sum(np.abs(self.channel_coeffs) ** 2))
        # Compute the frequency deviation and update freq_dev
        delta_f = -np.angle(np.mean(R_freq_shifted)) / (2 * np.pi * len(observedRes))
        self.freq_dev = self.freq_dev + self.lr * delta_f
if __name__ == "__main__":
    lengthToTake = 100000
    indToPlaceOne = 50000
    channel_coeeffs = np.array([0, 1,0,0,0])
    freq_dev = np.array([0.5])
    fs = 1
    channel = Channel(freq_dev, channel_coeeffs,fs, 1)
    sig = np.zeros([lengthToTake])
    sig[indToPlaceOne] = 1
    sigAfterChannel = channel.ApplyChanelOnSig(sig)
    recived = np.zeros([lengthToTake])
    recived[indToPlaceOne+1] = 1
    channel.Update_Channel_estimation(sig,recived)

    print(f"freq_dev = {np.array2string(channel.freq_dev)} and channel_coeffs = {np.array2string(channel.channel_coeffs)}")