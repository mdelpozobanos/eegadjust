"""
test_em
----------------------------------

Tests for `adjust_eeg.adjust_eeg.EM` class.
"""

from eegadjust.eegadjust import EM
import numpy as np
import pytest


@pytest.fixture(scope='module')
def data():
    """
    Randomly generates samples for two classes, one centered around 0 and
    one centered around 10.
    """

    # Some direct accessed for readability
    nrnd = np.random.normal

    # A controlled experiment will be created with N0 and N1 samples from
    # classes 0 and 1 respectively
    N0 = 20
    N1 = 4

    # Class 0 centered around 0
    X0 = nrnd(0, 1, N0)
    Y0 = np.zeros(N0)
    # Class 1 centered around 5
    X1 = nrnd(10, 1, N1)
    Y1 = np.ones(N1)

    # Concatenate into the sample space
    X = np.concatenate((X0, X1), axis=0)
    Y = np.concatenate((Y0, Y1), axis=0)

    return X, Y, [X0, X1]


@pytest.fixture(scope='module')
def em():
    """Returns an empty EM object"""
    return EM()


@pytest.fixture(scope='module')
def fitted_em(em, data):
    """Returns an EM object fitted with samples"""
    return em.fit(data[0])


class TestInstantiation:
    """Tests intantiation of a adjust_eeg.core.EM object"""

    def test_main(self):
        """Specifying initialization arguments"""
        EM(1., 1.)

    def test_main_no_args(self):
        """All initialization arguments are optional"""
        EM()
        EM(cost0=1.)
        EM(cost1=1.)

    def test_cost0(self):
        """cost0 must be a float > 0"""
        with pytest.raises(TypeError):
            EM(1)
        with pytest.raises(ValueError):
            EM(0.)

    def test_cost1(self):
        """cost1 must be a float > 0"""
        with pytest.raises(TypeError):
            EM(1., 1)
        with pytest.raises(ValueError):
            EM(1., 0.)


class TestNormX:
    """Tests the _norm_x method of EM"""

    def test_main(self, em):
        """x must be of type array-like"""
        assert(isinstance(em._norm_x([10, 20, 30]), np.ndarray))
        assert(isinstance(em._norm_x(np.zeros(10)), np.ndarray))

    def test_no_array(self, em):
        """x must be of type array-like"""
        with pytest.raises(ValueError):
            em._norm_x(10)

    def test_bad_vect(self, em):
        """x must be of a vector"""
        with pytest.raises(ValueError):
            em._norm_x(np.zeros([10, 10]))


class TestFit:
    """Tests the fit method of EM"""

    def test_main(self, em, data):
        """Fit x to the model"""
        x = data[0]
        x0 = data[2][0]
        x1 = data[2][1]
        em.fit(x)
        assert(np.isclose(em.means_[0], x0.mean(), 0.01))
        assert(np.isclose(em.means_[1], x1.mean(), 0.01))

    def test_x(self, em):
        """x must be a vector array"""
        with pytest.raises(ValueError):
            em.fit(10)
        with pytest.raises(ValueError):
            em.fit(np.zeros([10, 10]))


class TestPredict:
    """Tests the predict method of EM"""

    def test_main(self, fitted_em, data):
        """Predict labels"""
        x = data[0]
        y = data[1]
        py = fitted_em.predict(x)
        assert(all(py == y))

    def test_x(self, fitted_em):
        """x must be a vector array"""
        with pytest.raises(ValueError):
            fitted_em.predict(10)
        with pytest.raises(ValueError):
            fitted_em.predict(np.zeros([10, 10]))
