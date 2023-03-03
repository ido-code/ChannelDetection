import numpy as np

from Classes.Channel import Channel


class ChirpParams:
    def __init__(self, f_low, f_high, fs, time_to_transmit_chirp, time_between_chirps_begins, num_chirps):
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.time_to_transmit_chirp = time_to_transmit_chirp
        self.time_between_chirps_begins = time_between_chirps_begins
        self.num_chirps = num_chirps


class ChirpSigCreator:
    def __init__(self, chirp_params: ChirpParams, channel: Channel):
        self.chirp_params = chirp_params
        self.channel = channel
        assert self.chirp_params.fs == channel.fs, f"fs ({self.chirp_params.fs}) is not equal to channel.fs ({Channel.fs})"

    def create_one_chirp(self) -> np.ndarray:
        # Compute the time vector
        t = np.arange(0, self.chirp_params.time_to_transmit_chirp, 1 / self.chirp_params.fs)

        # Compute the instantaneous frequency of the chirp
        k = (self.chirp_params.f_high - self.chirp_params.f_low) / self.chirp_params.time_to_transmit_chirp
        inst_freq = self.chirp_params.f_low + k * t

        # Generate the chirp signal
        chirp_sig = np.exp(1j * 2 * np.pi * inst_freq * t)

        return chirp_sig

    def create_chirps_with_time_gaps(self) -> np.ndarray:
        # Create an empty array to hold the output signals
        num_samples = int(
            self.chirp_params.fs * self.chirp_params.time_between_chirps_begins * self.chirp_params.num_chirps)
        sig = np.zeros(num_samples)

        # Generate each chirp and insert it into the output array with the appropriate time gap
        chirp_sig = self.create_one_chirp()
        for i in range(self.chirp_params.num_chirps):
            start_index = int(i * self.chirp_params.fs * self.chirp_params.time_between_chirps_begins)
            end_index = start_index + len(chirp_sig)
            sig[start_index:end_index] = chirp_sig

        # Apply the channel to the signal
        sig_after_channel = self.channel.apply_chanel_on_sig(sig)

        return sig_after_channel
