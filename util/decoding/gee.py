#
# Written by: John Veillette 
#
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator
from sklearn.utils import indexable

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families
import statsmodels.api as sm

import numpy as np

FAMILIES = {

    # available GLM fmailies
    'binomial': families.Binomial,
    'gamma': families.Gamma,
    'gaussian': families.Gaussian,
    'inverse_gaussian': families.InverseGaussian,
    'negative_binomial': families.NegativeBinomial,
    'Poisson': families.Poisson,
    'Tweedie': families.Tweedie

}


class GEEEstimator(BaseEstimator):
    '''
    An sklearn-like wrapper for statsmodels's Generalized Estimating Equations
    with exchangeable within-group covariance. Great for e.g. predicting single
    trial variables for hold-out subjects, since it should be less prone to
    overfitting to the subjects in training set.

    A major drawback of GEEs is the lack of accepted goodness-of-fit tests
    for GEE-fit models, since GEE is a quasi-likelihood based method. To that
    end, assessing GEE models with cross-validation is a natural fit. However,
    statsmodels's API doesn't lend itself to repeated fitting or have any of
    the other bells and whistles you'd like in a prediction oriented setting,
    hence a scikit-learn wrapper so you get all that stuff for free without
    losing out on the valid parameter-level inference GEEs afford.

    Some notes (mostly to self):

    1.  Generalized estimating equations are a procedure used to fit GLMs even
        when the data have a heterogeneous (possibly unknown, but in this API
        we only consider grouped) covariance structure.

    2.  If you know something about the covariance structure, you can specify
        that when fitting the model, which will improve the efficiency of the
        estimator. In this class, we specify that residuals should be
        correlated within groups/subjects, but not between them. (This is also
        the premise of a GLMM, but GEE doesn't estimate the conditional/group
        parameters explicitly -- only the population parameters).

    3.  Parameter estimates are consistent (i.e. converge to the "true" values)
        even if the covariance structure is mispecified, but confidence metrics
        are only valid under such conditions **IF** you've used a robust
        covariance estimator. By default, statsmodels uses the Huber-White
        "sandwich" estimator as proposed by Liang and Zeger (1986), which is
        also the default in most packages such as R's gee. However, it performs
        poorly on grouped data when the number of groups is smaller than the
        number of observations per group, which is often the case for
        experimental data. You can instead use Mancl and DeRouen's (2001)
        bias-reduced estimator by setting cov_type = 'bias_reduced', or
        cov_type = 'naive', which will perform the best if the group structure
        in your data is correctly specified.

    4.  The GEE fitting procedure is semi-parametric, enforcing that the
        estimated model is linear but not assuming that the data come from
        a certain distribution. So you can have the partially-pooled goodness
        of a GLMM without the fragility of superfluous parametric assumptions.

    5.  Speaking of GLMMs, if you want to estimate group-level effects,
        you can't do that with GEEs. That's the price we pay for all the
        other goodies. So use this when you want to cross-validate across
        groups, but use GLMMs when you want to cross-validate within groups.

    6.  Since regularization is afforded by partial-pooling instead of by
        and L1/L2 penalty, and GEEs are naturally robust to overdispersion,
        we can compute p-values and confidence intervals for each coefficent,
        often without overfitting.

    '''

    def __init__(self, family = 'gaussian', cov_type = 'robust',
                    alpha = None, penalty_shape_param = 3.7,
                    oversample = False, class_weight = None, **family_kwargs):
        '''
        Arguments
        -----------

            family: an distribution family from statsmodels
                    that has NOT been initialized

            family_kwargs:  arguments to the family class constructor;
                            Can be used to change the link function from
                            the family's default if you want, but you might
                            run into issues with sklearn when using its
                            cross cross-validation or grid search utilities,
                            since it's clone() function has problems cloning
                            params which are external classes. Other arguments
                            are fine, and user-sepcified links are fine as
                            long as you don't need to clone.
            alpha: (float > 0) penalty weight. If None, no regularization term
                    is applied, which often works fine if your group structure
                    is correctly specified. Remember that, if regularization is
                    used, not all the inference methods (i.e. coefficient
                    p-values) are meaningful anymore. The regularization scheme
                    is SCAD as per [1] which can drive coefficients to zero,
                    so post-selection inference biases may occur if you take
                    p-values/ CIs at face value.
            penalty_shape_param: (float > 0) shape parameter for SCAD penalty
                                as per [1].
            oversample: (bool) for discrete y outcomes, will randomly oversample
                        from the minority class on fit

        Defaults to the first "safe" link function listed for each GLM
        distribution family, even if it's not the canonical link. For
        example, the canonical link for the gamma family is the inverse
        power link function, but inverse power doesn't respect the domain
        of the gamma distribution, causing numerical issues on some datasets.
        In contrast, the log link function won't cause any issues for gamma.
        This wrapper is meant to facilitate many sequential fits across many
        data partitions, where it may be impractical to babysit each
        model... so it always defaults to something that won't cause NaNs.

        If a regularizer is used, however, the canonical link is assumed!

        References
        ------------
        [1] Fan, J., & Li, R. (2001). Variable selection via nonconcave
            penalized likelihood and its oracle properties.
            Journal of the American statistical Association, 96(456), 1348-1360.

        Usage Example
        -----------

            import statsmodels.api as sm
            family = sm.families.Binomial # not initialized
            model = LinearEstimatingEquation(family)
            model.fit(X, y, groups) # use like sklearn model
        '''
        self.family = family
        self.family_kwargs = family_kwargs
        try:
            fam = self._family
        except:
            raise ValueError('Not a valid family. Try one of ' +
                                         ','.join(FAMILIES.keys()))
        self.cov_type = cov_type
        self.class_weight = class_weight
        self.alpha = alpha
        self.penalty_shape_param = penalty_shape_param
        self.oversample = oversample

    @property
    def _family(self):
        '''
        an initialized family class
        '''
        fam_name = self.family
        FamClass = FAMILIES[fam_name]
        if 'link' not in self.family_kwargs:
            # default to a link function that won't
            # cause numerical issues when fitting
            link = FamClass.safe_links[0]
            FamClass.links.append(link) # in case link class has alias
            self.family_kwargs['link'] = link
        family = FamClass(**self.family_kwargs)
        return family

    @property
    def _link(self):
        '''
        the link function (a statsmodels object)
        '''
        return self._family.link

    @property
    def _estimator_type(self):
        is_classifier = type(self._family) is families.Binomial
        est_type = 'classifier' if is_classifier else 'regressor'
        return est_type


    def fit(self, X, y, groups, **fit_params):
        '''
        Inputs
        -----------
        X: an (n_obs, n_predictors) array
        y: an (n_obs,) array compatible with GLM family
        groups: an (n_obs,) array of group IDs

        For **fit params, see
        statsmodels.genmod.generalized_estimating_equations.GEE
        Any argument to

        Outputs
        -----------
        self: (LinearEstimatingEquation) the fitted model

        '''
        # check inputs
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.ndim != 2:
            raise ValueError('only accepts 1 or 2-dimensional X, got '
                             '%s instead.' % (X.shape,))
        if y.ndim > 1:
            raise ValueError('only accepts up to 1-dimensional y, '
                             'got %s instead.' % (y.shape,))
        if groups.ndim > 2:
            raise ValueError('only accepts up to 2-dimensional groups, '
                             'got %s instead.' % (groups.shape,))

        if self.oversample:
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state = 0)
            X, y = sampler.fit_resample(X, y)
            groups = groups[sampler.sample_indices_]
            if 'weights' in fit_params:
                w = fit_params['weights']
                fit_params['weights'] = w[sampler.sample_indices_]

        assert(X.shape[0] == groups.shape[0])
        assert(y.shape[0] == X.shape[0])

        if self.class_weight is not None and 'weights' not in fit_params:
            weights = compute_sample_weight(self.class_weight, y = y)
        elif 'weights' in fit_params:
            weights = fit_params['weights']
            del fit_params['weights']
        else:
            weights = None

        if groups.ndim > 1: # multilevel group assignment
            _groups = groups[:, 0]
            dep_data = groups[:, 1:]
            cov_struct = sm.cov_struct.Nested()
        else: # only one level of group assignment
            _groups = groups
            dep_data = None
            cov_struct = sm.cov_struct.Exchangeable()

        # fit model
        _X = sm.add_constant(X, prepend = True) # manually adds intercept
        self.model_ = GEE(
            y, _X,
            groups = _groups,
            family = self._family,
            cov_struct = cov_struct, # group structure
            dep_data = dep_data,
            weights = weights,
            **fit_params
            )
        if self.alpha is None:
            self.results_ = self.model_.fit(
                cov_type = self.cov_type,
                **fit_params
                )
        else:
            self.results_ = self.model_.fit_regularized(
                                pen_wt = self.alpha,
                                scad_param = self.penalty_shape_param,
                                **fit_params
                                )
        return self

    def predict(self, X, **kwargs):
        '''
        Returns predictions for the input X. This is usually the parameter
        of the GLM distribution (e.g. rate parameter for Poisson family),
        but in the case of family == sm.families.Binomial(), it gives binary
        point predictions to remain consistent with sklearn LogisticRegression.

        If you want the predicted probability for a logistic or probit
        (a.k.a. binomial family) regression, use .predict_proba()

        Inputs
        -----------
        X: an (n_obs, n_predictors) array

        Outputs
        -----------
        y_pred: an (n_obs,) array with predictions
        '''
        if self.results_ is None:
            raise Exception("must fit model before predicting!")
        _X = sm.add_constant(X, prepend = True)
        yhat = self.results_.predict(_X, linear = False, **kwargs)
        if self._estimator_type == 'classifier':
            # return point prediction
            return (yhat > .5).astype(int)
        else:
            return yhat

    def predict_proba(self, X):
        '''
        only for family == sm.families.Binomial()

        Inputs
        -----------
        X: an (n_obs, n_predictors) array

        Outputs
        -----------
        y_prob: an (n_obs, 2) array with predicted probabilities of
                observation in X belonging to the negative & positive class
        '''
        if self._estimator_type != 'classifier':
            raise Exception("called predict_proba but y isn't binary!")
        if self.results_ is None:
            raise Exception("must fit model before predicting!")
        _X = sm.add_constant(X, prepend = True)
        yhat = self.results_.predict(_X, linear = False)
        return np.stack([1 - yhat, yhat], axis = 1)

    def predict_log_proba(self, X):
        y_prob = self.predict_proba(x)
        return np.log(y_prob)

    def decision_function(self, X, **kwargs):
        '''
        returns output of linear equation (prior to passing through inverse
        link function).
        '''
        if self.results_ is None:
            raise Exception("must fit model before predicting!")
        _X = sm.add_constant(X, prepend = True)
        return self.results_.predict(_X, linear = True, **kwargs)

    @property
    def coef_(self):
        '''
        (n_predictors,) array of model coefficients, exluding intercept
        '''
        if self.results_ is None:
            raise Exception("can't return coefficents until model is fit!")
        params = self.results_.params
        return params[1:] # exclude intercept

    @property
    def intercept_(self):
        '''
        just the intercept parameter (float)
        '''
        if self.results_ is None:
            raise Exception("can't return intercept until model is fit!")
        params = self.results_.params
        return params[0]

    def wald_test_full(self):
        '''
        tests full model against intercept-only model using naive parameter
        covariance estimate since robust covariance matrices are rank deficient
        for testing <n_predictor> restrictions at once

        Note that naive covariance estimator may yield misleading inferential
        statistics if the covariance structure is mispecified.
        Weight the results of this test accordingly.

        Outputs
        --------
        W: (float) chi2 distributed Wald statistic from test
        p_value: (float)
        '''
        # construct linear restriction matrix
        R = np.diag(np.ones(self.results_.params.size), 0)
        R = R[1:,:] # exclude intercept from restriction

        res = self.results_.wald_test(R, cov_p = self.results_.cov_naive)
        W = float(res.statistic[0][0])
        p = float(res.pvalue)
        return W, p

    def wald_test(self, mask):
        '''
        tests full model against nested model, specified by an
        (n_predictor,) boolean array where predictors you'd like to
        set to zero in restricted model are set to True

        Unlike .wald_test_full() method, this uses the covariance estimates
        that were used for model fitting. If parameter covariance
        matrix is rank deficient for the specified test, statsmodels
        will throw a warning, but you will be allowed to proceed.

        used to test whether the inclusion of a specified subset of
        predictors improves the model's fit

        Outputs
        --------
        W: (float) chi2 statistics from test
        p_value: (float)
        '''
        ## construct linear restriction matrix
        R = np.diag(np.ones(self.results_.params.size), 0)
        R = R[1:,:] # exclude intercept from restriction
        R = R[mask, :]

        res = self.results_.wald_test(R)
        W = float(res.statistic[0][0])
        p = float(res.pvalue)
        return W, p

    @property
    def p_values_(self):
        '''
        returns (uncorrected) two-sided p-values for predictor coefficents
        '''
        if self.results_ is None:
            raise Exception("model hasn't yet been fit!")
        assert(self.results_.use_t is False)
        return self.results_.pvalues[1:]

    @property
    def z_values_(self):
        '''
        returns z-values for predictor coefficents

        statsmodels for some reason uses the same attribute for t and z values,
        but GEEs in statsmodels use the Z statistic from a Wald test. We
        check the use_t attribute to make sure this is always true. (So these
        are the same z-values as would appear in statsmodels's summary table;
        I dug through their source code to make sure.)
        '''
        if self.results_ is None:
            raise Exception("model hasn't yet been fit!")
        assert(self.results_.use_t is False)
        return self.results_.tvalues[1:] # actually z-values

    @property
    def standard_errors_(self):
        '''
        returns standard errors for predictor coefficients
        '''
        if self.results_ is None:
            raise Exception("model hasn't yet been fit!")
        return self.results_.bse[1:]

    def get_confidence_interval(self, alpha = 0.05):
        '''
        returns an (n_predictor, 2) array with lower & upper bounds of
        a [100*(1 - alpha)]% confidence intervals for each coefficent
        '''
        if self.results_ is None:
            raise Exception("model hasn't yet been fit!")
        ci = self.results_.conf_int(alpha = alpha)
        return ci[1:, :]

    def transform(self, X):
        """Transform the data using the linear model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data to transform.
        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted targets.
        """
        return self.predict(X)

    def fit_transform(self, X, y, group, **fit_kwargs):
        """Fit the data and transform it using the linear model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples,)
            The target values.
        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted targets.
        """
        return self.fit(X, y, group,**fit_kwargs).transform(X)

    def score(self, X, y, **kwargs):
        '''
        deviance function of GLM distribution family
        '''
        if self._estimator_type == 'classifier':
            mu_hat = self.predict_proba(X)[:, 1]
        else:
            mu_hat = self.predict(X)
        return self._family.deviance(y, mu_hat, **kwargs)

    def summary(self, show = True, **kwargs):
        '''
        returns statsmodels.iolib.summary.Summary instance that can be
        printed or converted to various output formats
        '''
        if self.results_ is None:
            raise Exception("model hasn't yet been fit!")
        smry = self.results_.summary(**kwargs)
        return smry

    def get_pattern(self, mu):
        '''
        get the "pattern" that would predict an expected value of y,
        as per [1]. Latent signal is assumed to be uncorrelated
        with the noise term.

        Input
        --------
        mu: (float/int) expected value of y for which to compute encoding model

        Output
        --------
        pattern:    an (n_predictors,) array with an "activation" that
                    will predict E[y|x] = mu

        References
        ----------
        [1]     Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J. D.,
                Blankertz, B., & Bießmann, F. (2014). On the interpretation of
                weight vectors of linear models in multivariate neuroimaging.
                Neuroimage, 87, 96-110.
        '''
        s = self._link.inverse(mu) # latent signal magnitude that predicts mu

        # get X and estimated signal in matrix form (n_params, n_observations)
        X = self.model_.endog.T # don't exclude intercept like MNE does
        s_hat = self.decision_function(X.T[:, 1:]).T

        # encoding projection matrix, as per Remark 4 of [1]
        # (assuming s is 1d or s > 1d but uncorrelated)
        A = np.cov(X, s_hat)

        # for 1d latent signal, pattern is just s -> X projection x magnitude
        pattern = np.squeeze(A * s)
        return pattern[1:] # _now_ exclude intercept
