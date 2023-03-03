import numpy as np
from Channel import Channel

class ChirpParams:
    def __init__(self, f_low, f_high, fs, time_to_transmit_chirp, time_between_chirps_begins, num_chirps):
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.time_to_transmit_chirp = time_to_transmit_chirp
        self.time_between_chirps_begins = time_between_chirps_begins
        self.num_chirps = num_chirps


class ChirpSigCreator:
    def __init__(self, chirpparmas: ChirpParams, channel: Channel):
        self.chirpParams = chirpparmas
        self.channel = channel
        assert self.chirpparmas.fs == Channel.fs, f"fs ({self.chirpParams.fs}) is not equal to channel.fs ({Channel.fs})"

    def create_one_chirp(self):
        # Compute the time vector
        t = np.arange(0, self.chirpParams.time_to_transmit_chirp, 1 / self.chirpParams.fs)

        # Compute the instantaneous frequency of the chirp
        k = (self.chirpParams.f_high - self.chirpParams.f_low) / self.chirpParams.time_to_transmit_chirp
        inst_freq = self.chirpParams.f_low + k * t

        # Generate the chirp signal
        chirp_sig = np.exp(1j * 2 * np.pi * inst_freq * t)

        return chirp_sig

    def create_chirps_with_time_gaps(self) -> np.ndarray:
        # Create an empty array to hold the output signals
        num_samples = int(
            self.chirpParams.fs * self.chirpParams.time_between_chirps_begins * self.chirpParams.num_chirps)
        sig = np.zeros(num_samples)

        # Generate each chirp and insert it into the output array with the appropriate time gap
        chirp_sig = self.create_one_chirp()
        for i in range(self.chirpParams.num_chirps):
            start_index = int(i * self.chirpParams.fs * self.chirpParams.time_between_chirps_begins)
            end_index = start_index + len(chirp_sig)
            sig[start_index:end_index] = chirp_sig

        # Apply the channel to the signal
        sig_after_channel = self.channel.apply_chanel_on_sig(sig)

        return sig_after_channel
