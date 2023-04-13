import numpy as np
import pytest
import Classes.Channel


@pytest.mark.parametrize("channel_coeffs,freq_dev,expected_res",
                         [(np.array([0, 1]), np.array([0]), np.array([0, 1, 2, 3])),
                          [np.array([1]), np.array([0.25]), np.array([1, 2j, -3, 0])],
                          (np.array([0, 1]), np.array([0.5]), np.array([0, -1, 2, -3]))])
def test_only_coeffs(channel_coeffs, freq_dev, expected_res):
    fs = 1
    channel = Classes.Channel.Channel(freq_dev, channel_coeffs, fs)
    sig = np.array([1, 2, 3, 0])
    sig_after_channel = channel.apply_chanel_on_sig(sig)
    assert np.allclose(sig_after_channel, expected_res, 1e-5)


def test_update_channel():
    length_to_take = 100000
    ind_to_place_one = 50000
    np.random.seed(1234)  # set the seed for reproducibility

    mu = 0  # mean of the Gaussian distribution
    sigma = 1  # standard deviation of the Gaussian distribution
    n_samples = 10  # number of samples to generate

    channel_coeffs = np.random.normal(mu, sigma, n_samples)
    channel_coeffs = channel_coeffs / np.sqrt(np.sum(np.abs(channel_coeffs) ** 2))
    freq_dev = np.random.uniform(low=-0.5, high=0.5, size=1)
    fs = 1
    channel = Classes.Channel.Channel(freq_dev, channel_coeffs, fs, 1)
    sig = np.zeros([length_to_take])
    sig[ind_to_place_one] = 1
    received = channel.apply_chanel_on_sig(sig)
    channel.update_channel_estimation(sig, received)
    diff_between_res = channel.channel_coeffs - channel_coeffs
    max_diff = np.max(np.abs(diff_between_res))
    assert max_diff < 1e-4
    assert np.allclose(channel.freq_dev, freq_dev, 1e-4)
