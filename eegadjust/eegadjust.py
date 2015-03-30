# -*- coding: utf-8 -*-

"""
===============================
eegadjust.py
===============================

This is the main file from the eegadjust package.

"""

import copy
import numpy as np
import scipy.stats as sp_stats

# Dimensions are defined here for easy of interpretation of the code.
_c_dim = 0  # Component dimension
_t_dim = 1  # Time dimension
_ev_dim = 2  # Events dimension


def art_comp(bss_data, mix_mat, brain_areas, ch_dist, cost_c=1., cost_a=1.):
    """
    Automatic classification of EEG components computed by a BSS method.

    This function implements the classification part of the ADJUST algorithm
    described in [Mognon2010]_, which automatically divides the components in
    clean and artifactual groups.

    Parameters
    ----------
    bss_data : nupy.ndarray
        Array containing the time course of the BSS components. It must be a 3D
        array with dimensions CxTxE, where C, T and E are the number of
        components, time instants and recorded events respectively.
    mix_mat : numpy.ndarray
        Mixing matrix of the BSS algorithm with dimensions MxC, where M is
        the number of mixed signals and C the number of components.
    brain_areas : dict
        Dictionary with brain area clusters specified as logical vectors of
        length S. In particular, the following labels should be specified:
        *'eeg'* signaling strictly EEG channels,
        *'frontal'* signaling EEG frontal channels,
        *'posterior'* signaling EEG posterior channels,
        *'left-eye'* signaling EEG channels from the left eye, and
        *'right-eye'* signaling EEG channels from the right eye.
    ch_dist : numpy.ndarray
        Square matrix with dimensions SxS containing the distance between each
        pair of channels.
    cost_c : float, optional
        Cost of miss-classify a clean component as artifactual.
    cost_a : float, optional
        Cost of miss-classify an artifactual component as clean.

    Returns
    -------
    blink_comp : numpy.ndarray
        Logical vector of length C signaling components identified as blink
        artifacts.
    vert_comp : numpy.ndarray
        Logical vector of length C signaling components identified as vertical
        eye movement artifacts.
    horz_comp : numpy.ndarray
        Logical vector of length C signaling components identified as
        horizontal eye movement artifacts.
    disc_comp : numpy.ndarray
        Logical vector of length C signaling components identified as generic
        discontinuities artifacts.

    #-SPHINX-IGNORE-#
    References
    ----------
    [Mognon2010] A. Mognon, Jovicich J., Bruzzone L., and Buiatti M. Adjust: An
        automatic eeg artifact detector based on the joint use of spatial and
        temporal features. Psychophysiology, pages 229-240, July 2010.
    #-SPHINX-IGNORE-#
    """

    # Compute features
    gdsf_score = _gdsf(mix_mat, ch_dist, brain_areas)
    lre_lab = _lre(mix_mat, brain_areas)
    mev_score = _mev(bss_data)
    sad_score = _sad(mix_mat, brain_areas)
    sed_score = _sed(mix_mat, brain_areas)
    svd_lab = _svd(mix_mat, brain_areas)
    tk_score = _tk(bss_data)

    # Classify features using an EM classifier
    gdsf_lab = EM(cost0=cost_c,
                  cost1=cost_a).fit(gdsf_score).predict(gdsf_score)
    mev_lab = EM(cost0=cost_c, cost1=cost_a).fit(mev_score).predict(mev_score)
    sad_lab = EM(cost0=cost_c, cost1=cost_a).fit(sad_score).predict(sad_score)
    sed_lab = EM(cost0=cost_c, cost1=cost_a).fit(sed_score).predict(sed_score)
    tk_lab = EM(cost0=cost_c, cost1=cost_a).fit(tk_score).predict(tk_score)

    # Identify artifacts combining features
    blink_comp = tk_lab & sad_lab & lre_lab & svd_lab
    vert_comp = mev_lab & sad_lab & lre_lab & svd_lab
    horz_comp = mev_lab & sed_lab & ~lre_lab
    disc_comp = mev_lab & gdsf_lab

    # Return results
    return blink_comp, vert_comp, horz_comp, disc_comp


def _chk_parameters(num_comp=False, num_mix=False, mix_mat=None,
                    brain_areas=None, must_have_areas=None,
                    bss_data=None, ch_dist=None):
    """
    Checks input parameters.

    Parameters
    ----------
    num_comp : int, optional
        Number of BSS components.
    num_mix : int, optional
        Number of mixed signals.
    mix_mat : numpy.ndarray, optional
        Mixing matrix of the BSS algorithm with dimensions MxC, where M is
        the number of mixed signals and C the number of components.

        .. note:: The length of each dimension dimension can only be checked if
            *mix_mat* and *num_comp* are also specified.

        .. note:: If *mix_mat* and *num_comp* are not specified, the shape of
            *mix_mat* will be used as reference to extract such values.

    brain_areas : dict, optional
        Dictionary with brain area clusters specified as index vectors with
        length M. In particular, areas *must_have_areas* should be specified.

        .. note:: The range of the index vectors can only be checked if
            *mix_mat* or *num_mix* are also specified.

    must_have_areas : tuple of str, optional
        List of must-have brain areas.

        .. note:: *brain_areas* must be specified as well.

    bss_data : numpy.ndarray, optional
        3D array with the components temporal data. It must have dimensions
        CxTxE, where C, T and E are the number of component, time instants
        and events respectively.

        .. note:: The length of the first dimension can only be checked if
            *mix_mat* or *num_comp* are also specified.

    ch_dist : numpy.ndarray
        MxM matrix with the distance between channels.

    Returns
    -------
    result : {Exception, None}
        The detected exception if any. None otherwise.
    """

    if mix_mat is not None:
        if not isinstance(mix_mat, np.ndarray):
            return TypeError('mix_mat must be {}; is {} instead'
                             .format(type(np.zeros(0)), type(mix_mat)))
        if mix_mat.ndim != 2:
            return ValueError('mix_mat must be a 2D array; is {}D instead'
                              .format(mix_mat.ndim))
        # Extract/check the number of mixed signals and components
        if num_mix:
            assert num_mix == mix_mat.shape[0]
        else:
            num_mix = mix_mat.shape[0]
        if num_comp:
            assert num_comp == mix_mat.shape[1]
        else:
            num_comp = mix_mat.shape[1]

    if brain_areas is not None:
        # Now, check brain_areas
        if not isinstance(brain_areas, dict):
            return TypeError('brain_areas must be {}; is {} instead'
                             .format(type({}), type(brain_areas)))
        # Check each of the specified brain areas
        for area, value in brain_areas.iteritems():
            if not isinstance(value, np.ndarray):
                return TypeError('brain_areas values must be {}; '
                                 '{} is {} instead'
                                 .format(type(np.zeros(0)),
                                         area,
                                         type(value)))
            if value.ndim != 1:
                return ValueError('brain_areas values must be 1D; '
                                  '{} is {}D instead'.format(area, value.ndim))
            if value.dtype.type != np.int_:
                return TypeError('brain_areas values must be {} vectors; '
                                 '{} is a {} vector instead'
                                 .format(np.int_, area, value.dtype))
            if num_mix:
                if value.min() < -num_mix:
                    return ValueError('brain_areas values must be '
                                      'indices >= -M, with M the number of '
                                      'mixed signals; '
                                      '{} has minimum {} < -{} instead'
                                      .format(area, value.min(), num_mix))
                if value.max() >= num_mix:
                    return ValueError('brain_areas values must be '
                                      'indices < M, with M the number of '
                                      'mixed signals; '
                                      '{} has maximum {} >= {} instead'
                                      .format(area, value.max(), num_mix))
                # Assert that each value is unique
                aux = copy.copy(value)
                aux[aux < 0] = num_mix + aux[aux < 0]
                if len(np.unique(aux)) != len(value):
                    return ValueError('brain_areas values must not contain '
                                      'repeated values; {} does instead'
                                      .format(area))

    if must_have_areas is not None:
        # The following errors are risen, not returned, as they are code
        # errors, not parameter errors.

        # brain_areas should be specified as well.
        if brain_areas is None:
            raise KeyError('When must_have_areas is specified, brain_areas '
                           'should be specified as well.')
        # Now, check must_have_areas
        if not isinstance(must_have_areas, tuple):
            raise TypeError('must_have_areas must be {}; is {} instead'
                            .format(type(tuple()), type(must_have_areas)))
        for area in must_have_areas:
            if not isinstance(area, str):
                raise TypeError('must_have_areas values must be {}; '
                                'some are {} instead'
                                .format(type(''), type(area)))
            if area not in brain_areas:
                # This is the only returned error
                return ValueError('Key "{}" must be specified in brain_areas'
                                  .format(area))

    if bss_data is not None:
        if not isinstance(bss_data, np.ndarray):
            return TypeError('bss_data must be {}; is {} instead'
                             .format(type(np.zeros(0)), type(bss_data)))
        if bss_data.ndim != 3:
            return ValueError('bss_data must be 3D; is {}D instead'
                              .format(bss_data.ndim))
        if num_comp and bss_data.shape[0] != num_comp:
            return ValueError('bss_data must have dimensions MxTxE, '
                              'where M, T and E are the number of channels, '
                              'time instants and events respectively; '
                              'bss_data.shape[0] is {} != C = {} instead'
                              .format(bss_data.ndim, num_comp))

    if ch_dist is not None:
        if not isinstance(ch_dist, np.ndarray):
            return TypeError('ch_dist must be {}; is {} instead'
                             .format(type(np.zeros(0)), type(bss_data)))
        if ch_dist.ndim != 2:
            return ValueError('ch_dist must be 2D; is {}D instead'
                              .format(ch_dist.ndim))
        if num_mix and \
                (ch_dist.shape[0] != num_mix or ch_dist.shape[1] != num_mix):
            return ValueError('ch_dist must have dimensions MxM, where M is '
                              'the number of channels and time instants; '
                              'ch_dist.shape is {} != MxM = {}) instead'
                              .format(ch_dist.shape, (num_mix, num_comp)))


def _gdsf(mix_mat, ch_dist, brain_areas):
    """
    Generic Discontinuities Spatial Feature.

    Captures the spatial topography of generic discontinuities.

    Parameters
    ----------
    mix_mat : array
        Mixing matrix of the BSS algorithm with dimensions MxC, where M is
        the number of channels and C the number of components. This should
        only include strictly EEG channels (no EOG, MA, etc). If this is not
        the case, brain_areas parameter should be specified.
    ch_dist : array
        MxM matrix with the distance between channels.
    brain_areas : dict
        Dictionary with brain area clusters specified as index vectors of
        length S. In particular, label 'eeg' should be specified, signaling
        strictly EEG channels. If specified, EEG channels will be automatically
        selected from mix_mat and ch_dist.

    Returns
    -------
    res : array
        Vector of length C with the computed values for each component
    """

    # This feature is based only on EEG channels.
    try:
        eeg_ch = brain_areas['eeg']
    except TypeError:
        raise _chk_parameters(brain_areas=brain_areas)
    except KeyError:
        raise _chk_parameters(brain_areas=brain_areas,
                              must_have_areas=('eeg', ))
    num_ch = len(eeg_ch)
    try:
        eeg_mix_mat = mix_mat[eeg_ch]
    except TypeError:
        raise _chk_parameters(mix_mat=mix_mat)
    except IndexError:
        raise _chk_parameters(num_mix=mix_mat.shape[0],
                              brain_areas=brain_areas)

    try:
        eeg_ch_dist = ch_dist[eeg_ch][:, eeg_ch]
    except TypeError:
        raise _chk_parameters(ch_dist=ch_dist)
    except IndexError:
        raise _chk_parameters(num_mix=mix_mat.shape[0], ch_dist=ch_dist)

    # Decaying channel distance factor
    ch_f = np.exp(-eeg_ch_dist)
    # Only factors ij with i != j will be considered. Zero the diagonal
    for ch_n in range(num_ch):
        ch_f[ch_n, ch_n] = 0

    # Allocate temporal scores, with dimension #channels x #components
    try:
        score = np.zeros([num_ch, eeg_mix_mat.shape[1]])
    except IndexError:
        raise _chk_parameters(mix_mat=mix_mat)

    # Compute feature
    for ch_n in xrange(num_ch):
        # Consider factors of the 10 closest channels to ch_n
        close_ch = np.argsort(eeg_ch_dist[:, ch_n])[-10:]
        try:
            x = eeg_mix_mat[close_ch]*ch_f[close_ch][:, [ch_n, ]]
        except (ValueError, IndexError):
            raise _chk_parameters(mix_mat=mix_mat, ch_dist=ch_dist)
        score[ch_n, :] = eeg_mix_mat[ch_n] - np.mean(x, axis=0)

    # Return the maximum score for each component
    return score.max(axis=0)


def _lre(mix_mat, brain_areas):
    """
    Average IC topography weights across the left eye area.

    Captures the relationship between the activation of sources from left and
    right eyes areas. This features is used as control for blinks and vertical
    eye movements, and its inverse for horizontal eye movements.

    Parameters
    ----------
    mix_mat : array
        Mixing matrix of the BSS algorithm with dimensions SxC, where S is
        the number of sources and C the number of components.
    brain_areas : dict
        Dictionary with brain area clusters specified as logical vectors of
        length S. In particular, areas 'left-eye' and 'right-eye' should be
        specified.

    Returns
    -------
    res : array
        Boolean vector of length C with the computed flag for each component

    """
    try:
        left_eye = brain_areas['left-eye']
        right_eye = brain_areas['right-eye']
    except TypeError:
        raise _chk_parameters(brain_areas=brain_areas)
    except KeyError:
        raise _chk_parameters(brain_areas=brain_areas,
                              must_have_areas=('left-eye', 'right-eye'))

    try:
        res = np.sign(mix_mat[left_eye].mean(0)) == \
              np.sign(mix_mat[right_eye].mean(0))
    except TypeError:
        raise _chk_parameters(mix_mat=mix_mat)
    except IndexError:
        raise _chk_parameters(mix_mat=mix_mat, brain_areas=brain_areas)

    # mix_data dimensionality has to be checked explicitly, as a ND array with
    # N != 2 does not raise an exception
    if mix_mat.ndim != 2:
        raise _chk_parameters(mix_mat=mix_mat)

    # Done
    return res


def _mev(bss_data):
    """
    Maximum Epoch Variance.

    Captures the temporal dynamics of horizontal eye movements.

    Parameters
    ----------
    bss_data : array
        Array with dimensions CxTxE, where C is the number of components, T the
        number of time instants and E the number of events

    Returns
    -------
    res : array
        Vector of length C with the computed values for each component
    """
    # As _t_dim < _ev_dim, the events dimension will be shifted down one
    # position after computing the kurtosis (as time dimension will have
    # disappeared)
    ev_dim = _ev_dim - 1

    # Compute time variance
    try:
        var_data = bss_data.var(axis=_t_dim)
    except AttributeError:
        raise _chk_parameters(bss_data=bss_data)
    # Sort across events before trim
    try:
        np.sort(var_data, axis=ev_dim)
    except ValueError:
        raise _chk_parameters(bss_data=bss_data)
    # Trimmed vector of time variances
    var = sp_stats.trimboth(np.sort(var_data, axis=ev_dim),
                            0.01, axis=ev_dim)
    # Final results
    res = var.max(axis=ev_dim) / var.mean(axis=ev_dim)
    # bss_data dimensionality has to be checked explicitly, as a ND array with
    # N > 3 does not raise an exception
    if bss_data.ndim > 3:
        raise _chk_parameters(bss_data=bss_data)
    # Done
    return res


def _sad(mix_mat, brain_areas):
    """
    Spatial Average Difference.

    Captures spatial topography of blinks and vertical eye movements.

    Parameters
    ----------
    mix_mat : array
        Mixing matrix of the BSS algorithm with dimensions SxC, where S is
        the number of sources and C the number of components.
    brain_areas : dict
        Dictionary with brain area clusters specified as logical vectors of
        length S. In particular, areas 'frontal' and 'posterior' should be
        specified.

    Returns
    -------
    res : array
        Vector of length C with the computed values for each component
    """
    try:
        frontal = brain_areas['frontal']
        posterior = brain_areas['posterior']
    except TypeError:
        raise _chk_parameters(brain_areas=brain_areas)
    except KeyError:
        raise _chk_parameters(brain_areas=brain_areas,
                              must_have_areas=('frontal', 'posterior'))

    try:
        res = np.abs(mix_mat[frontal].mean(0)) - \
              np.abs(mix_mat[posterior].mean(0))
    except TypeError:
        raise _chk_parameters(mix_mat=mix_mat)
    except IndexError:
        raise _chk_parameters(mix_mat=mix_mat, brain_areas=brain_areas)

    # mix_data dimensionality has to be checked explicitly, as a ND array with
    # N != 2 does not raise an exception
    if mix_mat.ndim != 2:
        raise _chk_parameters(mix_mat=mix_mat)

    # Done
    return res


def _sed(mix_mat, brain_areas):
    """
    Spatial Eye Difference.

    Captures spatial topography of horizontal eye movements.

    Parameters
    ----------
    mix_mat : array
        Mixing matrix of the BSS algorithm with dimensions SxC, where S is
        the number of sources and C the number of components.
    brain_areas : dict
        Dictionary with brain area clusters specified as logical vectors of
        length S. In particular, areas 'left-eye' and 'right-eye' should be
        specified.

    Returns
    -------
    res : array
        Vector of length C with the computed values for each component
    """

    try:
        left_eye = brain_areas['left-eye']
        right_eye = brain_areas['right-eye']
    except TypeError:
        raise _chk_parameters(brain_areas=brain_areas)
    except KeyError:
        raise _chk_parameters(brain_areas=brain_areas,
                              must_have_areas=('left-eye', 'right-eye'))

    try:
        res = np.abs(mix_mat[left_eye].mean(0) -
                     mix_mat[right_eye].mean(0))
    except TypeError:
        raise _chk_parameters(mix_mat=mix_mat)
    except IndexError:
        raise _chk_parameters(mix_mat=mix_mat, brain_areas=brain_areas)

    # mix_data dimensionality has to be checked explicitly, as a ND array with
    # N != 2 does not raise an exception
    if mix_mat.ndim != 2:
        raise _chk_parameters(mix_mat=mix_mat)

    # Done
    return res


def _svd(mix_mat, brain_areas):
    """
    Spatial Variance Difference.

    Measures the variance difference between source activations of frontal and
    posterior areas. This feature is used as control for blinks and vertical
    eye movements.

    Parameters
    ----------
    mix_mat : array
        Mixing matrix of the BSS algorithm with dimensions SxC, where S is
        the number of sources and C the number of components.
    brain_areas : dict
        Dictionary with brain area clusters specified as logical vectors of
        length S. In particular, areas 'frontal' and 'posterior' should be
        specified.

    Returns
    -------
    res : array
        Vector of length C with the computed values for each component
    """
    try:
        frontal = brain_areas['frontal']
        posterior = brain_areas['posterior']
    except TypeError:
        raise _chk_parameters(brain_areas=brain_areas)
    except KeyError:
        raise _chk_parameters(brain_areas=brain_areas,
                              must_have_areas=('frontal', 'posterior'))

    try:
        res = mix_mat[frontal].mean(0).var(0) > \
              mix_mat[posterior].var(0)
    except TypeError:
        raise _chk_parameters(mix_mat=mix_mat)
    except IndexError:
        raise _chk_parameters(mix_mat=mix_mat, brain_areas=brain_areas)

    # mix_data dimensionality has to be checked explicitly, as a ND array with
    # N != 2 does not raise an exception
    if mix_mat.ndim != 2:
        raise _chk_parameters(mix_mat=mix_mat)

    # Done
    return res


def _tk(bss_data):
    """
    Temporal Kurtosis.

    Parameters
    ----------
    bss_data : array
        Array with dimensions CxTxE, where C is the number of components, T the
        number of time instants and E the number of events

    Returns
    -------
    res : array
        Vector of length C with the computed values for each component
    """
    # As _t_dim < _ev_dim, the events dimension will be shifted down one
    # position after computing the kurtosis (as time dimension will have
    # disappeared)
    ev_dim = _ev_dim - 1

    try:
        # Time kurtosis
        kurt_data = sp_stats.kurtosis(bss_data, axis=_t_dim)
        # Trimmed mean of the time kurtosis
        res = sp_stats.trim_mean(kurt_data, 0.01, ev_dim)
    except IndexError:
        raise _chk_parameters(bss_data=bss_data)
    # bss_data dimensionality has to be checked explicitly, as a ND array with
    # N > 3 does not raise an exception
    if bss_data.ndim > 3:
        raise _chk_parameters(bss_data=bss_data)
    return res


# ======================================================================================================================
# EXPECTATION MAXIMIZATION

class EM:
    """ Expectation Maximization.

    Unsupervised binary 1D classifier based on [Bruzzone2000]_.

    Parameters
    ----------
    cost0 : float
        Cost of missing of class 0
    cost1 : float
        Cost of missing of class 1

    Attributes
    ----------
    cost0 : float
        Cost of missing of class 0
    cost1 : float
        Cost of missing of class 1
    threshold : float
        Threshold maximizing the expectation of each distribution
    means\_ : list
        Mean of each class
    vars\_ : list
        Variance of each class
    priors\_ : list
        Prior-probability of each class

    References
    ----------
    .. [Bruzzone2000] L. Bruzzone and D.F. Prieto. Automatic analysis of the
        difference image for unsupervised change detection. IEEE trans. Geosci.
        Remote Sensing 38, 1171:1182, 2000.

    """

    means_ = [None, None]
    vars_ = [None, None]
    priors_ = [None, None]
    cost0 = None
    cost1 = None
    threshold = None

    def __init__(self, cost0=1., cost1=1.):
        # Check cost0
        if not isinstance(cost0, float):
            raise TypeError('cost0 must be of type float; '
                            'is {} instead'.format(type(cost0)))
        if not cost0 > 0:
            raise ValueError('cost0 must be greater than 0.')
        self.cost0 = cost0
        # Check cost1
        if not isinstance(cost1, float):
            raise TypeError('cost1 must be of type float; '
                            'is {} instead'.format(type(cost1)))
        if not cost1 > 0:
            raise ValueError('cost1 must be greater than 0.')
        self.cost1 = cost1

    def fit(self, x):
        """
        Fit the data to the model.

        Computes the threshold that maximizes the expectation of x when
        automatically divided in two classes.

        Parameters
        ----------
        x : array-like
            Data vector with length N, with N the number of x.

        """

        def bayes(mean, var, x):
            """
            Computes the probability of the given "x" values within a bayes
            distribution with the specified "mean" and "var".

            Parameters
            ----------
            mean : float
                Mean of the bayes distribution
            var : float
                Variance of the bayes distribution
            x : array-like
                Vector with the requested points

            Returns
            -------
            p : numpy.ndarray
                Vector of length = len(x) with the resulting probabilities
            """
            if var is 0:
                return 1
            else:
                return np.exp(((x-mean)**2)/(-2*var))/(np.sqrt(2*np.pi*var))

        # Normalize x
        x = self._norm_x(x)

        # Allocate variables
        num_samples = len(x)
        max_samples = x.max()
        min_samples = x.min()

        # Initialize classifier variables
        center = (max_samples + min_samples)/2.  # Central point
        alpha0 = 0.01*(center - min_samples)  # Threshold for class 0
        alpha1 = 0.01*(max_samples - center)  # Threshold for class 1

        # Expectation
        class0 = x[x < (center - alpha0)]  # Class 0 x
        class1 = x[x > (center + alpha1)]  # Class 1 x

        # Number of samples within each class
        num_class0 = float(len(class0))
        num_class1 = float(len(class1))

        # Each class mean, variance and prior probability
        mean0 = class0.mean()
        mean1 = class1.mean()
        var0 = class0.var()
        var1 = class1.var()
        prior0 = num_class0 / (num_class1 + num_class0)
        prior1 = num_class1 / (num_class1 + num_class0)

        # Allocate condition variables
        count = 0
        dif_mean_1 = 1  # difference between current and previous mean
        dif_mean_0 = 1
        dif_var_1 = 1  # difference between current and previous variance
        dif_var_0 = 1
        dif_prior_1 = 1  # difference between current and previous prior
        dif_prior_0 = 1

        # Stop criterion
        stop = 0.0001

        # [DBG] The following variables will be needed for debug plotting
        # prior0_vec = [prior0]
        # prior1_vec = [prior1]

        while (dif_mean_0 > stop) and (dif_mean_1 > stop) and \
                (dif_var_0 > stop) and (dif_var_1 > stop) and \
                (dif_prior_0 > stop) and (dif_prior_1 > stop) and \
                (count < 1000):

            count += 1

            # Copy of old values
            mean0_old = copy.copy(mean0)
            mean1_old = copy.copy(mean1)
            var0_old = copy.copy(var0)
            var1_old = copy.copy(var1)
            prior0_old = copy.copy(prior0)
            prior1_old = copy.copy(prior1)

            # Compute each sample prior probabilities from each class
            bayes0 = prior0_old*bayes(mean0_old, var0_old, x)
            bayes1 = prior1_old*bayes(mean1_old, var1_old, x)
            prior0_i = bayes0 / (bayes0 + bayes1)
            prior1_i = bayes1 / (bayes0 + bayes1)

            # Compute new prior, mean and var values
            prior0 = prior0_i.sum() / num_samples
            prior1 = prior1_i.sum() / num_samples
            mean0 = (prior0_i*x).sum() / (prior0*num_samples)
            mean1 = (prior1_i*x).sum() / (prior1*num_samples)
            var0 = (prior0_i*((x - mean0_old)**2)).sum() / (prior0*num_samples)
            var1 = (prior1_i*((x - mean1_old)**2)).sum() / (prior1*num_samples)

            # Compute the new loop conditions
            dif_mean_0 = np.abs(mean0 - mean0_old)
            dif_mean_1 = np.abs(mean1 - mean1_old)
            dif_var_0 = np.abs(var0 - var0_old)
            dif_var_1 = np.abs(var1 - var1_old)
            dif_prior_0 = np.abs(prior0 - prior0_old)
            dif_prior_1 = np.abs(prior1 - prior1_old)

            # [DBG] Some debugging plot
            # plt.ion()
            # prior0_vec.append(prior0)
            # prior1_vec.append(prior1)
            # plt.plot(range(count+1), prior0_vec, 'r',
            #          range(count+1), prior1_vec, 'b')
            # plt.xlim([0, 10*(1+count/10)])
            # plt.draw()
            # time.sleep(1)

        # Save means, variances and priors
        self.means_ = [mean0, mean1]
        self.vars_ = [var0, var1]
        self.priors_ = [prior0, prior1]
        # Thesholding
        k = float(self.cost0) / float(self.cost1)
        a = (var0 - var1) / 2
        b = (var1*mean0) - (var0*mean1)
        c = (var1*var0)*np.log((k*prior0*np.sqrt(var1)) / (prior1*np.sqrt(var0))) + \
            (var0*(mean1**2) - var1*(mean0**2))/2
        rad = (b**2) - (4*a*c)
        if rad < 0:
            raise ValueError('Negative discriminant encountered during '
                             'training!')

        soglia1 = (-b + np.sqrt(rad)) / (2*a)
        soglia2 = (-b - np.sqrt(rad)) / (2*a)

        if (soglia1 < mean0) or (soglia1 > mean1):
            self.threshold = soglia2
        else:
            self.threshold = soglia1

        if np.isnan(self.threshold):  # TO PREVENT CRASHES
            self.threshold = center

        # Done
        return self

    def predict(self, x):
        """
        Predicts the class of testing samples

        Parameters
        ----------
        x : array-like
            1D array with testing samples.

        Returns
        -------
        labels : numpy.ndarray
            Vector with resulting binary labels
        """
        # Normalize x parameter
        x = self._norm_x(x)
        # Apply threshold
        labels = np.zeros(len(x), int)
        labels[x > self.threshold] = 1
        # Done
        return labels

    def _norm_x(self, x):
        """
        Normalizes x to be of type np.ndarray

        Parameters
        ----------
        x : array-like
            1D array-like with samples.

        Returns
        -------
        arr : numpy.ndarray
            Vector array with samples
        """

        # Convert to an array if necessary
        if not isinstance(x, np.ndarray):
            arr = np.array(x)
        else:
            arr = x

        # arr dimension has to be explicitly checked, because the error
        # propagates too deep into the algorithm or doesn't raise
        if arr.ndim != 1:
            raise ValueError('x must be a vector (i.e. 1D); '
                             'is {}D instead'.format(arr.ndim))

        return arr
