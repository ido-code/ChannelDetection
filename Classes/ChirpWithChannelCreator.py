import numpy as np
from Channel import Channel

class ChirpSigCreator:
    def __init__(self, flow, fhigh, fs, timeToTransmitChirp, timeBetweenChirpsBegins, numChirps, channel_coeffs, freq_dev):
        self.flow = flow
        self.fhigh = fhigh
        self.fs = fs
        self.timeToTrasnitChirp = timeToTransmitChirp
        self.timeBetweenChirpsBegins = timeBetweenChirpsBegins
        self.numChirps = numChirps
        self.channel = Channel(freq_dev, channel_coeffs, fs)

    def CreateOneChirp(self):
        # Compute the time vector
        t = np.arange(0, self.timeToTrasnitChirp, 1 / self.fs)

        # Compute the instantaneous frequency of the chirp
        k = (self.fhigh - self.flow) / self.timeToTrasnitChirp
        inst_freq = self.flow + k * t

        # Generate the chirp signal
        chirp_sig = np.exp(1j * 2 * np.pi * inst_freq * t)

        return chirp_sig

    def CreateChirpsWithTimeGaps(self):
        # Create an empty array to hold the output signals
        num_samples = int(self.fs *  self.timeBetweenChirpsBegins * self.numChirps)
        sig = np.zeros(num_samples)

        # Generate each chirp and insert it into the output array with the appropriate time gap
        chirp_sig = self.CreateOneChirp()
        for i in range(self.numChirps):
            start_index = int(i * self.fs * self.timeBetweenChirpsBegins)
            end_index = start_index + len(chirp_sig)
            sig[start_index:end_index] = chirp_sig

        # Apply the channel to the signal
        sig_after_channel = self.channel.ApplyChanelOnSig(sig)

        return sig_after_channel