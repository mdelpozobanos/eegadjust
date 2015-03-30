#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_eegadjust
----------------------------------

Tests for `eegadjust` module.
"""

import pytest
import numpy as np
from eegadjust import eegadjust


@pytest.fixture(scope='module')
def data():
    """
    Randomly generates data
    """

    # Some direct accessed for readability
    urnd = np.random.uniform
    nrnd = np.random.normal

    # A controlled experiment will be created with C components containing L
    # time samples and E events.
    C = 100
    L = 1000
    E = 10

    # Time vector
    t = np.linspace(0, 10, E*L)

    # Generator: Original components
    comp_fcn = lambda n: urnd(-1, 1, 1)*np.cos(urnd(0, 1, 1)*n) + \
                         urnd(-1, 1, 1)*np.sin(urnd(0, 1, 1)*n)
    # Generator: Added noise
    noise_fcn = lambda n: np.abs(nrnd(1, 0.1, len(n))**4) * \
                          np.cos(urnd(0, 0.5, 1)*n)

    # Generate components
    bss_data = []
    for c_n in range(C):
        bss_data.append(comp_fcn(t) + noise_fcn(t))

    # Merge into an array
    bss_data = np.array(bss_data)  # Noisy components (stat point)
    # Reshape components
    bss_data = bss_data.reshape([C, L, E])

    # Generate a mixing matrix
    mix_mat = urnd(-5, 5, [C, C])

    # Generate brain areas dictionary
    true_false = lambda n1, n2: np.concatenate(
        (np.zeros(n1, dtype=bool),
         np.ones(n2 - n1, dtype=bool),
         np.zeros(C - n2, dtype=bool)),
        1)
    brain_areas = {'eeg': np.where(true_false(0, 95))[0],
                   'frontal': np.where(true_false(0, 30))[0],
                   'posterior': np.where(true_false(50, 90))[0],
                   'left-eye': np.where(true_false(5, 10))[0],
                   'right-eye': np.where(true_false(10, 15))[0]}

    # Generate distance between channels
    aux = np.triu(nrnd(0, 2, [C, C]), 1)
    ch_dist = aux + aux.T
    del aux

    # Return only the first 90 components
    return {'bss_data': bss_data[:90], 'mix_mat': mix_mat[:, :90],
            'brain_areas': brain_areas, 'ch_dist': ch_dist}


class TestChkParameters:
    """Test _chk_parameters function"""

    # (mix_mat=None, brain_areas=None, must_have_areas=None, bss_data=None)

    def test_mix_mat(self, data):
        """*mix_mat* must be a 2D numpy.ndarray"""
        _test_parameters(eegadjust._chk_parameters, 'mix_mat')
        # Dimensions can also be specified
        eegadjust._chk_parameters(num_mix=4, num_comp=2,
                                  mix_mat=np.zeros([4, 2]))

    def test_bss_data(self, data):
        """*bss_data* must be a 3D numpy.ndarray"""
        _test_parameters(eegadjust._chk_parameters, 'bss_data')

    def test_mix_bss_congruence(self):
        """*bss_data* must be a 3-D array"""
        _test_parameters(eegadjust._chk_parameters, 'mix_bss')

    def test_ch_dist(self):
        """*ch_dist* must be a 2-D array"""
        _test_parameters(eegadjust._chk_parameters, 'ch_dist')

    def test_mix_dist_congruence(self):
        """*ch_dist* must have shape NxN with N the number of sources"""
        _test_parameters(eegadjust._chk_parameters, 'mix_dist')

    def test_brain_areas(self, data):
        """*brain_areas* must be a dict"""
        _test_parameters(eegadjust._chk_parameters, 'brain_areas')

        # Each entry must have valid indices (within the range)
        with pytest.raises(ValueError) as err:
            res = eegadjust._chk_parameters(
                brain_areas={'bad_area': np.array([10, 0, 4])},
                num_mix=10)
            if res is not None:
                raise res
        assert 'brain_areas values must be indices < M' in err.value.message
        with pytest.raises(ValueError) as err:
            res = eegadjust._chk_parameters(
                brain_areas={'bad_area': np.array([-11, 0, 4])},
                num_mix=10)
            if res is not None:
                raise res
        assert 'brain_areas values must be indices >= -M' in err.value.message

        # Each entry must have unique indices
        with pytest.raises(ValueError) as err:
            res = eegadjust._chk_parameters(
                brain_areas={'bad_area': np.array([9, -1])},
                num_mix=10)
            if res is not None:
                raise res
        assert 'brain_areas values must not contain' in err.value.message

        # When a required brain area is not available, an exception is risen
        with pytest.raises(ValueError) as err:
            res = eegadjust._chk_parameters(
                brain_areas={'a': np.array([0, 1, 2])},
                must_have_areas=('b', )
            )
            if res is not None:
                raise res
        assert 'must be specified in brain_areas' in err.value.message

    def test_must_have_areas(self, data):
        """*must_have_areas* must be a tuple of strings"""

        # Correct usage
        eegadjust._chk_parameters(brain_areas=data['brain_areas'],
                                  must_have_areas=tuple('eeg'))
        res = eegadjust._chk_parameters(brain_areas=data['brain_areas'],
                                        must_have_areas=tuple('error'))
        assert isinstance(res, ValueError)

        # Must be a tuple
        with pytest.raises(TypeError) as err:
            eegadjust._chk_parameters(brain_areas=data['brain_areas'],
                                      must_have_areas='eeg')
        assert 'must_have_areas must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            eegadjust._chk_parameters(brain_areas=data['brain_areas'],
                                      must_have_areas=['eeg'])
        assert 'must_have_areas must be <' in err.value.message

        # Must be a tuple of strings
        with pytest.raises(TypeError) as err:
            eegadjust._chk_parameters(brain_areas=data['brain_areas'],
                                      must_have_areas=('eeg', 0))
        assert 'must_have_areas values must be <' in err.value.message

        # Should be specified along with brain_areas
        with pytest.raises(KeyError) as err:
            eegadjust._chk_parameters(must_have_areas=tuple('eeg'))
        assert 'When must_have_areas is specified,' in err.value.message


class TestGDSF:
    """Tests _gdsf function"""

    def test_main(self, data):
        """
        Checks the computation of generic discontinuities spatial features
        """
        res = eegadjust._gdsf(data['mix_mat'],
                              data['ch_dist'],
                              data['brain_areas'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_mix_mat(self, data):
        """*mix_mat* must be a 2-D numpy.ndarray"""
        _test_parameters(eegadjust._gdsf, 'mix_mat',
                         ch_dist=data['ch_dist'],
                         brain_areas=data['brain_areas'])

    def test_ch_dist(self, data):
        """*ch_dist* must be a 2-D numpy.ndarray"""
        _test_parameters(eegadjust._gdsf, 'ch_dist',
                         mix_mat=data['mix_mat'],
                         brain_areas=data['brain_areas'])

    def test_brain_areas(self, data):
        """*brain_areas* must be a dict"""
        _test_parameters(eegadjust._gdsf, 'brain_areas', must_have=('eeg', ),
                         mix_mat=data['mix_mat'],
                         ch_dist=data['ch_dist'])
        # Area "eeg" must be defined
        with pytest.raises(ValueError) as err:
            eegadjust._gdsf(data['mix_mat'], data['ch_dist'], {})
        assert 'Key "eeg" must be specified' in err.value.message


class TestMEV:
    """Tests _mev function"""

    def test_main(self, data):
        """Checks the computation of the maximum epoch variance"""
        res = eegadjust._mev(data['bss_data'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_bss_data(self, data):
        """*bss_data* must be a 3D numpy.ndarray"""
        _test_parameters(eegadjust._mev, 'bss_data')


class TestTK:
    """Tests _tk function"""

    def test_main(self, data):
        """Checks the computation of the temporal kurtosis"""
        res = eegadjust._tk(data['bss_data'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_bss_data(self):
        """*bss_data* must be a 3D numpy.ndarray"""
        _test_parameters(eegadjust._tk, 'bss_data')


@pytest.fixture(scope='class',
                params=[eegadjust._lre, eegadjust._sad, eegadjust._sed,
                        eegadjust._svd],
                ids=['_lre', '_sad', '_sed', '_svd'])
def fcn(request):
    return request.param


class TestFcn():
    """
    Tests functions with the common signature fcn_name(mix_mat, brain_areas)
    """

    def test_main(self, data, fcn):
        """Checks the proper call"""
        res = fcn(data['mix_mat'], data['brain_areas'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_mix_mat(self, data, fcn):
        """*mix_mat* must be a 2-D matrix"""
        _test_parameters(fcn, 'mix_mat', brain_areas=data['brain_areas'])

    def test_brain_areas(self, data, fcn):
        """*brain_areas* must be a dict"""
        if fcn in (eegadjust._lre, eegadjust._sed):
            must_have = ('left-eye', 'right-eye')
        else:
            must_have = ('frontal', 'posterior')
        _test_parameters(fcn, 'brain_areas', must_have=must_have,
                         mix_mat=data['mix_mat'])
        # Area "eeg" must be defined
        with pytest.raises(ValueError) as err:
            fcn(data['mix_mat'], {})
        assert 'must be specified' in err.value.message


class TestArtComp:
    """Test function eegadjust.art_comp"""

    def test_main(self, data):
        """Test the identification of artifactual components"""
        eegadjust.art_comp(data['bss_data'], data['mix_mat'],
                           data['brain_areas'], data['ch_dist'])

    def test_bss_data(self, data):
        """*bss_data* must be a 3D numpy.ndarray"""
        _test_parameters(eegadjust.art_comp, 'bss_data',
                         mix_mat=data['mix_mat'],
                         brain_areas=data['brain_areas'],
                         ch_dist=data['ch_dist'])

    def test_mix_mat(self, data):
        """*mix_mat* must be a 2-D np.ndarray"""
        _test_parameters(eegadjust.art_comp, 'mix_mat',
                         bss_data=data['bss_data'],
                         brain_areas=data['brain_areas'],
                         ch_dist=data['ch_dist'])

    def test_brain_areas(self, data):
        """*brain_areas* must be a dict"""
        _test_parameters(eegadjust.art_comp, 'brain_areas',
                         must_have=data['brain_areas'].keys(),
                         bss_data=data['bss_data'],
                         mix_mat=data['mix_mat'],
                         ch_dist=data['ch_dist'])

    def test_ch_dist(self, data):
        """*ch_dist* must be a 2-D np.ndarray"""
        _test_parameters(eegadjust.art_comp, 'ch_dist',
                         bss_data=data['bss_data'],
                         mix_mat=data['mix_mat'],
                         brain_areas=data['brain_areas'])


def _test_parameters(fcn, key, must_have=None, *args, **kwargs):
    """
    Tests function parameters.

    This function asserts that a wrongly specified input parameter is detected
    by a function

    Parameters
    ----------
    fcn : functions
        Function to be called
    key : str
        Name of the parameter to be tested
    *args : tuple
        Non-keyword arguments passed to the function, other than "key"
    **kwargs : dict
        Keyword arguments passed to the function, other than "key"
    """

    if key is 'bss_data':
        # Must be a numpy.ndarray
        with pytest.raises(TypeError) as err:
            res = fcn(bss_data=0., *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(bss_data='error', *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(bss_data=[0, 10], *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be <' in err.value.message

        # Must be 3D
        with pytest.raises(ValueError) as err:
            res = fcn(bss_data=np.zeros([4]*2), *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError) as err:
            res = fcn(bss_data=np.zeros([4]*4), *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be 3D' in err.value.message

    elif key is 'mix_bss':
        # mix_mat's and bss_data's shapes must be congruent
        with pytest.raises(ValueError) as err:
            res = fcn(bss_data=np.zeros([4]*3), mix_mat=np.zeros([4, 2]),
                      *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'bss_data must have dimensions MxTxE' in err.value.message

    elif key is 'mix_mat':

        # Must be a numpy.ndarray
        with pytest.raises(TypeError) as err:
            res = fcn(mix_mat=0., *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(mix_mat='error', *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be <' in err.value.message
        # Must be 2D
        with pytest.raises(ValueError) as err:
            res = fcn(mix_mat=np.zeros([100]*1), *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be a 2D array' in err.value.message
        with pytest.raises(ValueError) as err:
            res = fcn(mix_mat=np.zeros([100]*3), *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be a 2D array' in err.value.message

    elif key is 'brain_areas':

        # Must be a dict
        with pytest.raises(TypeError) as err:
            res = fcn(brain_areas='error', *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'brain_areas must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(brain_areas=0, *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'brain_areas must be <' in err.value.message

        # Each entry must be a 1D numpy.ndarray
        with pytest.raises(TypeError) as err:
            res = fcn(brain_areas={'bad_area': 0}, *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'brain_areas values must be <' in err.value.message
        with pytest.raises(ValueError) as err:
            res = fcn(brain_areas={'bad_area': np.zeros([2]*2)},
                      *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'brain_areas values must be 1D' in err.value.message

        # Each entry must be an array of integers
        with pytest.raises(TypeError) as err:
            res = fcn(brain_areas={'bad_area': np.array([0., 0.3, 0.54])},
                      *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'brain_areas values must be <' in err.value.message

        # The following tests are only for methods other than _chk_parameters
        if fcn is not eegadjust._chk_parameters:
            brain_areas = {}
            for mh in must_have:
                brain_areas[mh] = np.array([0, 1])
            # Each entry must have valid indices (within the range)
            brain_areas[must_have[0]] = np.array([5000, 0, 4])
            with pytest.raises(ValueError) as err:
                fcn(brain_areas=brain_areas, *args, **kwargs)
            assert 'brain_areas values must be indices <' in err.value.message
            brain_areas[must_have[0]] = np.array([-5000, 0, 4])
            with pytest.raises(ValueError) as err:
                fcn(brain_areas=brain_areas, *args, **kwargs)
            assert 'brain_areas values must be indices >=' in err.value.message

            # Each entry must have unique indices
            # num_mix = kwargs['mix_mat'].shape[0]
            # with pytest.raises(ValueError) as err:
            #     res = fcn(brain_areas={must_have: np.array([num_mix-1, -1])},
            #               *args, **kwargs)
            #     if fcn is eegadjust._chk_parameters and res is not None:
            #         raise res
            # assert 'brain_areas values must not contain' in err.value.message

            # When a required brain area is available, an exception is risen
            with pytest.raises(ValueError) as err:
                fcn(brain_areas={'a': np.array([0, 1, 2])}, *args, **kwargs)
            assert 'must be specified in brain_areas' in err.value.message

    elif key is 'ch_dist':
        # Must be a numpy.ndarray
        with pytest.raises(TypeError) as err:
            res = fcn(ch_dist='error', *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'ch_dist must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(ch_dist=[0, 0, 0], *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'ch_dist must be <' in err.value.message
        # Must be a 2D array
        with pytest.raises(ValueError) as err:
            res = fcn(ch_dist=np.zeros(100), *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'ch_dist must be 2D' in err.value.message
        with pytest.raises(ValueError) as err:
            res = fcn(ch_dist=np.zeros([100]*3), *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'ch_dist must be 2D' in err.value.message

    elif key is 'mix_dist':

        # mix_mat's and bss_data's shapes must be congruent
        with pytest.raises(ValueError) as err:
            res = fcn(ch_dist=np.zeros([4]*2), mix_mat=np.zeros([3, 4]),
                      *args, **kwargs)
            if fcn is eegadjust._chk_parameters and res is not None:
                raise res
        assert 'ch_dist must have dimensions MxM' in err.value.message

    else:
        raise SystemError('Test Code Error! '
                          'Specified option {} is not supported.'
                          .format(key))
