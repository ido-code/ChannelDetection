import numpy as np
import Classes.Channel
def test_Only_Coeffs():
    channel_coeeffs = np.array([0,1])
    freq_dev = np.array([0])
    fs = 1
    channel = Classes.Channel.Channel(freq_dev, channel_coeeffs,fs)
    sig = np.array([1,2,3,0])
    sigAfterChannel = channel.apply_chanel_on_sig(sig)
    expectedRes = np.array([0,1,2,3])
    assert np.allclose(sigAfterChannel,expectedRes,1e-5)
def test_Only_freq_dev():
    channel_coeeffs = np.array([1])
    freq_dev = np.array([0.25])
    fs = 1
    channel = Classes.Channel.Channel(freq_dev, channel_coeeffs,fs)
    sig = np.array([1,2,3,0])
    sigAfterChannel = channel.apply_chanel_on_sig(sig)
    expectedRes = np.array([1,2j,-3,0])
    assert np.allclose(sigAfterChannel,expectedRes,1e-5)
def test_All_Channel():
    channel_coeeffs = np.array([0,1])
    freq_dev = np.array([0.5])
    fs = 1
    channel = Classes.Channel.Channel(freq_dev, channel_coeeffs,fs)
    sig = np.array([1,2,3,0])
    sigAfterChannel = channel.apply_chanel_on_sig(sig)
    expectedRes = np.array([0,-1,2,-3])
    assert np.allclose(sigAfterChannel,expectedRes,1e-5)

def test_update_channel():
    lengthToTake = 100000
    indToPlaceOne = 50000
    np.random.seed(1234)  # set the seed for reproducibility

    mu = 0  # mean of the Gaussian distribution
    sigma = 1  # standard deviation of the Gaussian distribution
    n_samples = 10  # number of samples to generate

    channel_coeeffs = np.random.normal(mu, sigma, n_samples)
    channel_coeeffs = channel_coeeffs/ np.sqrt(np.sum(np.abs(channel_coeeffs) ** 2))
    freq_dev = np.random.uniform(low=-0.5, high=0.5, size=1)
    fs = 1
    channel = Classes.Channel.Channel(freq_dev, channel_coeeffs, fs, 1)
    sig = np.zeros([lengthToTake])
    sig[indToPlaceOne] = 1
    recived = channel.apply_chanel_on_sig(sig)
    channel.update_channel_estimation(sig, recived)
    diff_between_res = channel.channel_coeffs - channel_coeeffs
    max_diff = np.max(np.abs(diff_between_res))
    assert max_diff < 1e-4
    assert np.allclose(channel.freq_dev,freq_dev,1e-4)