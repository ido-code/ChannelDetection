import numpy as np
from unittest.mock import Mock, patch
from pytest import mark
from Classes.Channel import Channel
from Classes.ChirpWithChannelCreator import ChirpSigCreator, ChirpParams

ind_to_start = lambda fs,t,ind_reps: int(fs*t*ind_reps)

def setup_expected_result(fs, time_between_chirps_begins, num_chirps, sig_to_rep):
    num_samples = int(fs*time_between_chirps_begins*num_chirps)
    output_sig = np.zeros([num_samples])
    for ind in range(num_chirps):
        start_ind = int(ind*fs*time_between_chirps_begins)
        end_ind = start_ind + len(sig_to_rep)
        output_sig[start_ind:end_ind] = sig_to_rep
    return  output_sig
@mark.parametrize("f_low, f_high, fs, time_to_transmit_chirp, time_between_chirps_begins, num_chirps", [
    (100, 1000, 44100, 0.1, 0.5, 10),
    (200, 2000, 48000, 0.05, 1, 5),
])
def test_create_chirps_with_time_gaps(f_low, f_high, fs, time_to_transmit_chirp, time_between_chirps_begins, num_chirps):
    # Mock the channel object
    channel = Mock(spec=Channel)
    channel.fs = fs
    # Set up the chirp params
    chirp_params = ChirpParams(f_low, f_high, fs, time_to_transmit_chirp, time_between_chirps_begins, num_chirps)

    # Mock the channel's apply_chanel_on_sig method to return the mock signal
    sig_to_rep = np.array([1])
    # Create a mock output signal
    channel.apply_chanel_on_sig = lambda x:  x
    # Create an instance of the chirp signal creator
    creator = ChirpSigCreator(chirp_params, channel)

    # Call the create_chirps_with_time_gaps method and check the result
    with patch.object(ChirpSigCreator, 'create_one_chirp', return_value=sig_to_rep):
        result = creator.create_chirps_with_time_gaps()

    # Check that the channel's apply_chanel_on_sig method was called with the expected argument
    expected_sig = setup_expected_result(fs, time_between_chirps_begins, num_chirps, sig_to_rep)
    assert np.allclose(result, expected_sig)

    assert channel.apply_chanel_on_sig.call_args[0][0].shape == expected_sig.shape
    assert np.allclose(channel.apply_chanel_on_sig.call_args[0][0], expected_sig)


def test_create_one_chirp():
    # Set up the chirp params
    chirp_params = ChirpParams(100, 1000, 44100, 0.1, 0.5, 10)
    channel_mock = Mock()
    channel_mock.fs = chirp_params.fs
    # Create an instance of the chirp signal creator
    creator = ChirpSigCreator(chirp_params, channel_mock)

    # Call the create_one_chirp method and check the result
    result = creator.create_one_chirp()
    assert result.shape == (4410,)

    # Check that the chirp's frequency increases over time
    k = (chirp_params.f_high - chirp_params.f_low) / chirp_params.time_to_transmit_chirp
    t = np.arange(0, chirp_params.time_to_transmit_chirp, 1 / chirp_params.fs)
    inst_freq = chirp_params.f_low + k * t
    diff = np.max(np.abs((np.pi + np.angle(result) - 2 * np.pi * inst_freq * t) % (2*np.pi) - np.pi))
    assert diff < 1e-5
