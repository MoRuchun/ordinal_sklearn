"""Ordinal regression models and utilities following a scikit-learn style API.

This module implements a small research-oriented framework for ordinal regression.
It provides three classical ordinal model families (cumulative link, sequential,
adjacent category) that can leverage any scikit-learn-compatible regressor as a
latent function approximator through a coordinate descent routine reminiscent of
OGBoost.  Additional utility functions simplify dataset loading from Excel files,
model evaluation, fragility curve visualization, and hyperparameter search.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit, logsumexp
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
    column_or_1d,
    has_fit_parameter,
)


_EPS = 1e-12


def _check_link_function(name: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return CDF and PDF callables for the requested link function.

    Parameters
    ----------
    name:
        Either ``"probit"`` or ``"logit"``. Case-insensitive.

    Returns
    -------
    cdf, pdf:
        A tuple of callables implementing the cumulative distribution function
        and corresponding probability density function.
    """

    key = name.lower()
    if key not in {"probit", "logit", "logistic"}:
        raise ValueError("link must be either 'probit' or 'logit'.")

    if key == "probit":
        return norm.cdf, norm.pdf

    def logistic_pdf(z: np.ndarray) -> np.ndarray:
        sig = expit(z)
        return sig * (1.0 - sig)

    return expit, logistic_pdf


class BaseOrdinalRegressor(BaseEstimator, ClassifierMixin, ABC):
    """Abstract parent class for ordinal regression models.

    The class implements a coordinate descent routine that alternates between
    functional gradient updates of latent regressors and monotonic threshold
    optimization.  Sub-classes must provide model-specific likelihood,
    probability, and gradient calculations.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_classes: int,
        *,
        max_iter: int = 200,
        tol: float = 1e-5,
        learning_rate: float = 0.1,
        link: str = "probit",
        threshold_learning_rate: float = 0.1,
        max_line_search_iter: int = 20,
        random_state: Optional[int] = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.link = link
        self.threshold_learning_rate = threshold_learning_rate
        self.max_line_search_iter = max_line_search_iter
        self.random_state = random_state

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Sequence[int], sample_weight: Optional[np.ndarray] = None) -> "BaseOrdinalRegressor":
        """Fit the ordinal regression model.

        Parameters
        ----------
        X:
            Training feature matrix of shape ``(n_samples, n_features)``.
        y:
            Ordinal class labels. Labels are mapped onto ``0, …, K-1``
            internally while preserving the original ordering for
            predictions.
        sample_weight:
            Optional non-negative weights applied to each training sample.

        Returns
        -------
        self:
            Fitted estimator.
        """

        X, y = check_X_y(X, y, accept_sparse=False, dtype=float)
        y = column_or_1d(y, warn=True)
        if np.any(np.isnan(X)):
            raise ValueError("Input features must not contain NaNs.")

        self._rng = check_random_state(self.random_state)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != self.n_classes:
            raise ValueError(
                "n_classes does not match the number of unique labels provided."
            )

        # Map arbitrary ordered labels onto 0..K-1 while preserving ordering.
        sort_index = np.argsort(self.classes_)
        self.classes_ = self.classes_[sort_index]
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([class_to_index[val] for val in y], dtype=int)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("Ordinal regression requires at least two classes.")

        self._cdf, self._pdf = _check_link_function(self.link)
        self._n_latent = self._n_latent_functions()
        n_samples = X.shape[0]

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight must be a vector of length n_samples")

        # Initialize latent representation and threshold parameterization.
        self._latent_train_ = np.zeros((n_samples, self._n_latent), dtype=float)
        self._base_models_: List[List[Tuple[BaseEstimator, float]]] = [
            [] for _ in range(self._n_latent)
        ]
        self._alpha_ = 0.0
        if self.n_classes_ > 2:
            self._delta_ = np.zeros(self.n_classes_ - 2, dtype=float)
        else:
            self._delta_ = np.zeros(0, dtype=float)

        thresholds = self._thresholds_from_parameters(self._alpha_, self._delta_)
        nll = self._negative_log_likelihood(y_encoded, self._latent_train_, thresholds, sample_weight)
        self.nll_history_: List[float] = [nll]

        for iteration in range(self.max_iter):
            # Latent function update (f-step)
            grad_latent = self._gradient_latent(y_encoded, self._latent_train_, thresholds, sample_weight)
            self._update_latent_functions(X, sample_weight, grad_latent, y_encoded)

            # Threshold update (tau-step)
            thresholds = self._thresholds_from_parameters(self._alpha_, self._delta_)
            grad_tau = self._gradient_thresholds(
                y_encoded, self._latent_train_, thresholds, sample_weight
            )
            self._alpha_, self._delta_, thresholds = self._update_thresholds(
                y_encoded, sample_weight, grad_tau, thresholds
            )

            new_nll = self._negative_log_likelihood(
                y_encoded, self._latent_train_, thresholds, sample_weight
            )
            self.nll_history_.append(new_nll)

            if iteration > 0:
                rel_change = abs(self.nll_history_[-2] - new_nll) / max(
                    self.nll_history_[-2], 1e-8
                )
                if rel_change < self.tol:
                    break

        self.thresholds_ = thresholds
        self.X_train_ = X
        self.y_train_ = y_encoded
        self.sample_weight_ = sample_weight
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ordinal class labels for ``X`` using maximum probability."""

        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for ``X``."""

        check_is_fitted(self, "thresholds_")
        X = check_array(X, accept_sparse=False, dtype=float)
        latent = self._predict_latent_functions(X)
        thresholds = self._thresholds_from_parameters(self._alpha_, self._delta_)
        return self._predict_proba_from_latent(latent, thresholds)

    def score(self, X: np.ndarray, y: Sequence[int]) -> float:
        """Compute accuracy score for the provided test data."""

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    # ------------------------------------------------------------------
    # Helper infrastructure shared by subclasses
    # ------------------------------------------------------------------
    def _thresholds_from_parameters(self, alpha: float, delta: np.ndarray) -> np.ndarray:
        """Convert unconstrained parameters into strictly increasing thresholds."""

        if self.n_classes_ == 2:
            return np.array([alpha], dtype=float)
        thresholds = np.empty(self.n_classes_ - 1, dtype=float)
        thresholds[0] = alpha
        for idx in range(1, self.n_classes_ - 1):
            thresholds[idx] = thresholds[idx - 1] + math.exp(delta[idx - 1])
        return thresholds

    def _predict_latent_functions(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the latent function(s) for arbitrary input samples."""

        if not hasattr(self, "_base_models_"):
            raise ValueError("Model must be fitted before calling predict_proba.")
        latent = np.zeros((X.shape[0], self._n_latent), dtype=float)
        for idx in range(self._n_latent):
            for estimator, weight in self._base_models_[idx]:
                latent[:, idx] += weight * estimator.predict(X)
        return latent

    def _update_latent_functions(
        self,
        X: np.ndarray,
        sample_weight: np.ndarray,
        grad_latent: np.ndarray,
        y_encoded: np.ndarray,
    ) -> None:
        """Perform a functional gradient step for each latent regressor."""

        for idx in range(self._n_latent):
            residual = -grad_latent[:, idx]
            mask = self._latent_mask(idx, y_encoded)
            if mask is None:
                mask = np.ones_like(y_encoded, dtype=bool)
            mask = mask.astype(bool)
            if not np.any(mask):
                continue

            estimator = clone(self.base_estimator)
            X_sub = X[mask]
            residual_sub = residual[mask]
            if sample_weight is None:
                sw_sub = None
            else:
                sw_sub = sample_weight[mask]

            if sw_sub is not None and not has_fit_parameter(estimator, "sample_weight"):
                estimator.fit(X_sub, residual_sub)
            else:
                estimator.fit(X_sub, residual_sub, sample_weight=sw_sub)

            update = estimator.predict(X)
            self._latent_train_[:, idx] += self.learning_rate * update
            self._base_models_[idx].append((estimator, self.learning_rate))

    def _update_thresholds(
        self,
        y_encoded: np.ndarray,
        sample_weight: np.ndarray,
        grad_tau: np.ndarray,
        thresholds: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Update threshold parameters with a constrained line search."""

        grad_alpha, grad_delta = self._project_threshold_gradient(grad_tau)
        base_params = np.concatenate(([self._alpha_], self._delta_))
        gradient = np.concatenate(([grad_alpha], grad_delta))
        if np.linalg.norm(gradient, ord=np.inf) < 1e-12:
            return self._alpha_, self._delta_, thresholds

        step = self.threshold_learning_rate
        current_nll = self._negative_log_likelihood(
            y_encoded, self._latent_train_, thresholds, sample_weight
        )
        params_new = base_params.copy()
        for _ in range(self.max_line_search_iter):
            candidate = base_params - step * gradient
            alpha_new = candidate[0]
            delta_new = candidate[1:]
            thresholds_new = self._thresholds_from_parameters(alpha_new, delta_new)
            nll_new = self._negative_log_likelihood(
                y_encoded, self._latent_train_, thresholds_new, sample_weight
            )
            if nll_new <= current_nll:
                params_new = candidate
                current_nll = nll_new
                step *= 2.0
                break
            step *= 0.5
        alpha_final = params_new[0]
        delta_final = params_new[1:]
        thresholds_final = self._thresholds_from_parameters(alpha_final, delta_final)
        return alpha_final, delta_final, thresholds_final

    def _project_threshold_gradient(self, grad_tau: np.ndarray) -> Tuple[float, np.ndarray]:
        """Map gradients with respect to thresholds onto unconstrained parameters."""

        if self.n_classes_ == 2:
            return float(np.sum(grad_tau)), np.zeros(0, dtype=float)

        g_alpha = float(np.sum(grad_tau))
        g_delta = np.zeros_like(self._delta_)
        cumulative = 0.0
        for idx in reversed(range(self.n_classes_ - 1)):
            cumulative += grad_tau[idx]
            if idx > 0:
                g_delta[idx - 1] = cumulative * math.exp(self._delta_[idx - 1])
        return g_alpha, g_delta

    # ------------------------------------------------------------------
    # Abstract hooks implemented by subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _n_latent_functions(self) -> int:
        """Return the number of latent functions managed by the model."""

    @abstractmethod
    def _negative_log_likelihood(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        """Compute the weighted negative log-likelihood."""

    @abstractmethod
    def _gradient_latent(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        """Gradient of the negative log-likelihood with respect to latent values."""

    @abstractmethod
    def _gradient_thresholds(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        """Gradient of the negative log-likelihood with respect to thresholds."""

    @abstractmethod
    def _predict_proba_from_latent(
        self, latent: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Map latent values and thresholds to class probabilities."""

    def _latent_mask(self, idx: int, y_encoded: np.ndarray) -> Optional[np.ndarray]:
        """Return a boolean mask describing which samples update latent ``idx``."""

        return np.ones_like(y_encoded, dtype=bool)


class CumulativeOrdinalRegressor(BaseOrdinalRegressor):
    """Cumulative link ordinal regression model.

    This model corresponds to the proportional odds / probit formulation with
    monotonic thresholds.  It reduces to binary logistic/probit regression when
    ``n_classes == 2``.
    """

    def _n_latent_functions(self) -> int:  # noqa: D401
        return 1

    def _negative_log_likelihood(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        f = latent[:, 0]
        n = f.shape[0]
        probs = self._predict_proba_from_latent(latent, thresholds)
        chosen = probs[np.arange(n), y_encoded]
        chosen = np.clip(chosen, _EPS, 1.0)
        loss = -np.sum(sample_weight * np.log(chosen))
        return loss / np.sum(sample_weight)

    def _predict_proba_from_latent(
        self, latent: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        f = latent[:, 0]
        n = f.shape[0]
        K = self.n_classes_
        if K == 2:
            cdf = self._cdf(thresholds[0] - f)
            probs = np.column_stack([cdf, 1.0 - cdf])
            return np.clip(probs, _EPS, 1.0)

        diff = thresholds[np.newaxis, :] - f[:, np.newaxis]
        cdf = self._cdf(diff)
        probs = np.empty((n, K), dtype=float)
        probs[:, 0] = cdf[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = cdf[:, k] - cdf[:, k - 1]
        probs[:, -1] = 1.0 - cdf[:, -1]
        probs = np.clip(probs, _EPS, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _gradient_latent(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        f = latent[:, 0]
        n = f.shape[0]
        K = self.n_classes_
        diff = thresholds[np.newaxis, :] - f[:, np.newaxis]
        cdf = self._cdf(diff)
        pdf = self._pdf(diff)
        probs = self._predict_proba_from_latent(latent, thresholds)
        grad = np.zeros(n, dtype=float)

        for i in range(n):
            k = y_encoded[i]
            p = probs[i, k]
            if K == 2:
                pdf_val = pdf[i, 0]
                if k == 0:
                    dp_df = -pdf_val
                else:
                    dp_df = pdf_val
            else:
                if k == 0:
                    dp_df = -pdf[i, 0]
                elif k == K - 1:
                    dp_df = pdf[i, -1]
                else:
                    dp_df = -pdf[i, k] + pdf[i, k - 1]
            grad[i] = -sample_weight[i] * dp_df / np.clip(p, _EPS, None)

        return grad[:, np.newaxis]

    def _gradient_thresholds(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        f = latent[:, 0]
        diff = thresholds[np.newaxis, :] - f[:, np.newaxis]
        pdf = self._pdf(diff)
        probs = self._predict_proba_from_latent(latent, thresholds)
        grad = np.zeros_like(thresholds)
        K = self.n_classes_

        for i in range(f.shape[0]):
            k = y_encoded[i]
            weight = sample_weight[i]
            if K == 2:
                p = probs[i, k]
                if k == 0:
                    grad[0] += -weight * pdf[i, 0] / np.clip(p, _EPS, None)
                else:
                    grad[0] += weight * pdf[i, 0] / np.clip(p, _EPS, None)
                continue

            p = probs[i, k]
            if k == 0:
                grad[0] += -weight * pdf[i, 0] / np.clip(p, _EPS, None)
            elif k == K - 1:
                grad[-1] += weight * pdf[i, -1] / np.clip(p, _EPS, None)
            else:
                grad[k] += -weight * pdf[i, k] / np.clip(p, _EPS, None)
                grad[k - 1] += weight * pdf[i, k - 1] / np.clip(p, _EPS, None)

        return grad


class SequentialOrdinalRegressor(BaseOrdinalRegressor):
    """Sequential (continuation-ratio) ordinal regression model.

    The model implements a continuation-ratio likelihood where each binary task
    corresponds to predicting whether the response exceeds a particular ordinal
    threshold.  Category-specific latent functions are supported through
    independent ensembles for each binary sub-problem.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_classes: int,
        *,
        category_specific: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(base_estimator, n_classes, **kwargs)
        self.category_specific = category_specific

    def _n_latent_functions(self) -> int:  # noqa: D401
        if self.category_specific:
            return self.n_classes_ - 1
        return 1

    def _latent_mask(self, idx: int, y_encoded: np.ndarray) -> Optional[np.ndarray]:
        if not self.category_specific:
            return np.ones_like(y_encoded, dtype=bool)
        return y_encoded >= idx

    def _negative_log_likelihood(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        probs = self._predict_proba_from_latent(latent, thresholds)
        chosen = probs[np.arange(len(y_encoded)), y_encoded]
        chosen = np.clip(chosen, _EPS, 1.0)
        loss = -np.sum(sample_weight * np.log(chosen))
        return loss / np.sum(sample_weight)

    def _predict_proba_from_latent(
        self, latent: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        n = latent.shape[0]
        K = self.n_classes_
        if self.category_specific:
            f = latent
        else:
            f = np.repeat(latent[:, :1], K - 1, axis=1)

        diff = thresholds[np.newaxis, :] - f
        q = self._cdf(diff)
        q = np.clip(q, _EPS, 1.0 - _EPS)
        probs = np.zeros((n, K), dtype=float)
        if K == 2:
            probs[:, 0] = 1.0 - q[:, 0]
            probs[:, 1] = q[:, 0]
            return np.clip(probs, _EPS, 1.0)

        prod_q = np.cumprod(q, axis=1)
        prev_prod = np.concatenate([np.ones((n, 1)), prod_q[:, :-1]], axis=1)
        probs[:, 0] = 1.0 - q[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = prev_prod[:, k] - prod_q[:, k]
        probs[:, -1] = prod_q[:, -1]
        probs = np.clip(probs, _EPS, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _gradient_latent(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        n = latent.shape[0]
        K = self.n_classes_
        if self.category_specific:
            f = latent
        else:
            f = np.repeat(latent[:, :1], K - 1, axis=1)
        diff = thresholds[np.newaxis, :] - f
        q = np.clip(self._cdf(diff), _EPS, 1.0 - _EPS)
        pdf = self._pdf(diff)
        probs = self._predict_proba_from_latent(latent, thresholds)

        grad = np.zeros_like(f)
        for i in range(n):
            k = y_encoded[i]
            p = probs[i, k]
            prod_prefix = np.ones(K - 1)
            for m in range(K - 1):
                if m > 0:
                    prod_prefix[m] = prod_prefix[m - 1] * q[i, m - 1]
            prod_full = prod_prefix * q[i]
            for m in range(K - 1):
                dq_df = -pdf[i, m]
                if k == 0:
                    if m == 0:
                        dp_df = dq_df * (-1)
                    else:
                        dp_df = 0.0
                elif k == K - 1:
                    if m < K - 1:
                        dp_dq = prod_full[m] / q[i, m]
                        dp_df = dp_dq * dq_df
                    else:
                        dp_df = 0.0
                else:
                    if m < k:
                        dp_dq = probs[i, k] / q[i, m]
                        dp_df = dp_dq * dq_df
                    elif m == k:
                        dp_dq = -prod_prefix[m]
                        dp_df = dp_dq * dq_df
                    else:
                        dp_df = 0.0
                grad[i, m if self.category_specific else 0] += -sample_weight[i] * dp_df / np.clip(p, _EPS, None)
        if not self.category_specific:
            return grad[:, :1]
        return grad

    def _gradient_thresholds(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        n = latent.shape[0]
        K = self.n_classes_
        if self.category_specific:
            f = latent
        else:
            f = np.repeat(latent[:, :1], K - 1, axis=1)
        diff = thresholds[np.newaxis, :] - f
        q = np.clip(self._cdf(diff), _EPS, 1.0 - _EPS)
        pdf = self._pdf(diff)
        probs = self._predict_proba_from_latent(latent, thresholds)

        grad = np.zeros_like(thresholds)
        for i in range(n):
            k = y_encoded[i]
            p = probs[i, k]
            prod_prefix = np.ones(K - 1)
            for m in range(K - 1):
                if m > 0:
                    prod_prefix[m] = prod_prefix[m - 1] * q[i, m - 1]
            prod_full = prod_prefix * q[i]
            for m in range(K - 1):
                dq_dtau = pdf[i, m]
                if k == 0:
                    if m == 0:
                        dp_dtau = dq_dtau * (-1)
                    else:
                        dp_dtau = 0.0
                elif k == K - 1:
                    dp_dq = prod_full[m] / q[i, m]
                    dp_dtau = dp_dq * dq_dtau
                else:
                    if m < k:
                        dp_dq = probs[i, k] / q[i, m]
                        dp_dtau = dp_dq * dq_dtau
                    elif m == k:
                        dp_dq = -prod_prefix[m]
                        dp_dtau = dp_dq * dq_dtau
                    else:
                        dp_dtau = 0.0
                grad[m] += -sample_weight[i] * dp_dtau / np.clip(p, _EPS, None)
        return grad


class AdjacentCategoryOrdinalRegressor(BaseOrdinalRegressor):
    """Adjacent category ordinal regression model."""

    def _n_latent_functions(self) -> int:  # noqa: D401
        return 1

    def _negative_log_likelihood(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        probs = self._predict_proba_from_latent(latent, thresholds)
        chosen = probs[np.arange(len(y_encoded)), y_encoded]
        chosen = np.clip(chosen, _EPS, 1.0)
        loss = -np.sum(sample_weight * np.log(chosen))
        return loss / np.sum(sample_weight)

    def _predict_proba_from_latent(
        self, latent: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        f = latent[:, 0]
        n = f.shape[0]
        K = self.n_classes_
        if K == 2:
            diff = thresholds[0] - f
            a = np.clip(self._cdf(diff), _EPS, 1.0 - _EPS)
            probs = np.column_stack([a, 1.0 - a])
            return np.clip(probs, _EPS, 1.0)

        diff = thresholds[np.newaxis, :] - f[:, np.newaxis]
        a = np.clip(self._cdf(diff), _EPS, 1.0 - _EPS)
        s = np.zeros((n, K), dtype=float)
        for j in range(1, K):
            s[:, j] = s[:, j - 1] + np.log1p(-a[:, j - 1]) - np.log(a[:, j - 1])
        log_norm = logsumexp(s, axis=1)
        probs = np.exp(s - log_norm[:, np.newaxis])
        probs = np.clip(probs, _EPS, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _gradient_latent(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        f = latent[:, 0]
        n = f.shape[0]
        K = self.n_classes_
        diff = thresholds[np.newaxis, :] - f[:, np.newaxis]
        a = np.clip(self._cdf(diff), _EPS, 1.0 - _EPS)
        pdf = self._pdf(diff)
        s = np.zeros((n, K), dtype=float)
        ds_df = np.zeros((n, K), dtype=float)
        if K > 2:
            for j in range(1, K):
                s[:, j] = s[:, j - 1] + np.log1p(-a[:, j - 1]) - np.log(a[:, j - 1])
                ratio = pdf[:, j - 1] / (a[:, j - 1] * (1.0 - a[:, j - 1]))
                ds_df[:, j] = ds_df[:, j - 1] + ratio
        log_norm = logsumexp(s, axis=1)
        probs = np.exp(s - log_norm[:, np.newaxis])

        weighted = (probs * ds_df).sum(axis=1)
        grad = -sample_weight * (ds_df[np.arange(n), y_encoded] - weighted)
        return grad[:, np.newaxis]

    def _gradient_thresholds(
        self,
        y_encoded: np.ndarray,
        latent: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        f = latent[:, 0]
        n = f.shape[0]
        K = self.n_classes_
        diff = thresholds[np.newaxis, :] - f[:, np.newaxis]
        a = np.clip(self._cdf(diff), _EPS, 1.0 - _EPS)
        pdf = self._pdf(diff)
        s = np.zeros((n, K), dtype=float)
        ds_dtau = np.zeros((n, K, K - 1), dtype=float)
        if K > 2:
            for j in range(1, K):
                s[:, j] = s[:, j - 1] + np.log1p(-a[:, j - 1]) - np.log(a[:, j - 1])
                for m in range(K - 1):
                    ds_dtau[:, j, m] = ds_dtau[:, j - 1, m]
                idx = j - 1
                ds_dtau[:, j, idx] += -pdf[:, idx] / (a[:, idx] * (1.0 - a[:, idx]))
        log_norm = logsumexp(s, axis=1)
        probs = np.exp(s - log_norm[:, np.newaxis])

        grad = np.zeros(K - 1, dtype=float)
        for m in range(K - 1):
            term = (probs * ds_dtau[:, :, m]).sum(axis=1)
            grad[m] = -np.sum(
                sample_weight
                * (
                    ds_dtau[np.arange(n), y_encoded, m]
                    - term
                )
            )
        return grad


# ----------------------------------------------------------------------
# Dataset utilities
# ----------------------------------------------------------------------

def load_ordinal_excel(
    path: str,
    target_column: str,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> Dict[str, np.ndarray]:
    """Load and standardize an ordinal regression dataset from Excel.

    Parameters
    ----------
    path:
        Path to the ``.xlsx`` file.
    target_column:
        Name of the column containing the ordinal response variable.
    test_size:
        Fraction of the data reserved for the test split.
    val_size:
        Fraction of the *remaining* training data reserved for validation.
    random_state:
        Optional random seed for the data splits.
    stratify:
        If ``True`` (default) perform stratified splits using the response
        labels.

    Returns
    -------
    dict
        A dictionary containing ``X_train``, ``X_val``, ``X_test``, ``y_train``,
        ``y_val``, ``y_test``, and the fitted ``scaler``.
    """

    df = pd.read_excel(path)
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset.")

    y = df[target_column].to_numpy()
    X = df.drop(columns=[target_column]).to_numpy(dtype=float)

    stratify_labels = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    stratify_val = y_train if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_val,
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
    }


# ----------------------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------------------

def _brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Compute the multiclass Brier score."""

    n_samples, n_classes = proba.shape
    y_onehot = np.zeros((n_samples, n_classes), dtype=float)
    y_onehot[np.arange(n_samples), y_true] = 1.0
    return np.mean(np.sum((proba - y_onehot) ** 2, axis=1))


def evaluate_models(
    models: Dict[str, BaseOrdinalRegressor],
    data: Dict[str, np.ndarray],
    *,
    plot_confusion: bool = False,
    confusion_cmap: str = "viridis",
) -> pd.DataFrame:
    """Evaluate ordinal models on train/validation/test splits.

    Parameters
    ----------
    models:
        Mapping from human-readable model names to fitted ordinal estimators.
    data:
        Dictionary containing the splits produced by :func:`load_ordinal_excel`.
    plot_confusion:
        If ``True`` draw a normalized confusion matrix heatmap for the test set.
    confusion_cmap:
        Matplotlib colormap used for the confusion matrix visualization.

    Returns
    -------
    pandas.DataFrame
        Table of evaluation metrics sorted by test accuracy.
    """

    required_keys = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
    missing = required_keys.difference(data)
    if missing:
        raise KeyError(f"Missing required dataset keys: {missing}")

    rows = []
    for name, model in models.items():
        metrics: Dict[str, float] = {}
        for split in ("train", "val", "test"):
            X_split = data[f"X_{split}"]
            y_split = np.asarray(data[f"y_{split}"])
            proba = model.predict_proba(X_split)
            pred_idx = np.argmax(proba, axis=1)
            y_pred = model.classes_[pred_idx]
            y_indices = np.searchsorted(model.classes_, y_split)
            metrics[f"Accuracy ({split})"] = accuracy_score(y_split, y_pred)
            metrics[f"Macro-F1 ({split})"] = f1_score(
                y_split, y_pred, average="macro"
            )
            metrics[f"QWK ({split})"] = cohen_kappa_score(
                y_split, y_pred, weights="quadratic"
            )
            metrics[f"Brier ({split})"] = _brier_score(y_indices, proba)
            metrics[f"NLL ({split})"] = log_loss(
                y_indices, proba, labels=np.arange(model.n_classes_)
            )
        row = {"Model": name}
        row.update(metrics)
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values(by="Accuracy (test)", ascending=False)
    results = results.reset_index(drop=True)

    if plot_confusion:
        best_model_name = results.iloc[0]["Model"]
        best_model = models[best_model_name]
        proba = best_model.predict_proba(data["X_test"])
        pred_indices = np.argmax(proba, axis=1)
        y_pred = best_model.classes_[pred_indices]
        conf_mat = pd.crosstab(
            pd.Series(data["y_test"], name="True"),
            pd.Series(y_pred, name="Predicted"),
            normalize="index",
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_mat, annot=True, fmt=".2f", cmap=confusion_cmap)
        plt.title(f"Normalized Confusion Matrix ({best_model_name})")
        plt.tight_layout()

    return results


# ----------------------------------------------------------------------
# Fragility curve visualization
# ----------------------------------------------------------------------

def plot_fragility_curves(
    model: BaseOrdinalRegressor,
    baseline_features: np.ndarray,
    varying_feature_index: int,
    varying_values: np.ndarray,
    *,
    mode: str = "cumulative",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot fragility curves for a trained ordinal model.

    Parameters
    ----------
    model:
        Fitted ordinal regression estimator.
    baseline_features:
        Reference feature vector used to populate non-varying predictors.
    varying_feature_index:
        Index of the feature that will be swept across ``varying_values``.
    varying_values:
        Monotonic grid of feature values for the fragility plot.
    mode:
        Either ``"cumulative"`` to show ``P(Y >= k)`` or ``"category"`` to show
        the individual category probabilities.
    ax:
        Optional Matplotlib axis; if ``None`` a new figure and axis are created.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the fragility curves.
    """

    check_is_fitted(model, "thresholds_")
    baseline = np.array(baseline_features, dtype=float)
    X = np.tile(baseline, (len(varying_values), 1))
    X[:, varying_feature_index] = varying_values
    proba = model.predict_proba(X)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if mode == "cumulative":
        cumulative = np.flip(np.cumsum(np.flip(proba, axis=1), axis=1), axis=1)
        for idx in range(1, model.n_classes_):
            ax.plot(varying_values, cumulative[:, idx], label=f"P(Y ≥ {model.classes_[idx]})")
    elif mode == "category":
        for idx in range(model.n_classes_):
            ax.plot(varying_values, proba[:, idx], label=f"P(Y = {model.classes_[idx]})")
    else:
        raise ValueError("mode must be either 'cumulative' or 'category'.")

    ax.set_xlabel(f"Feature {varying_feature_index}")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    return ax


# ----------------------------------------------------------------------
# Hyperparameter search utility
# ----------------------------------------------------------------------

def hyperparameter_search(
    estimator_cls: Callable[..., BaseOrdinalRegressor],
    param_grid: Dict[str, Iterable],
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "grid",
    n_iter: int = 25,
    cv: int = 3,
    scoring: str = "accuracy",
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Perform hyperparameter search for ordinal estimators.

    Parameters
    ----------
    estimator_cls:
        Constructor for the ordinal model (e.g., :class:`CumulativeOrdinalRegressor`).
    param_grid:
        Hyperparameter search space expressed as a mapping from parameter names
        to candidate values.
    X, y:
        Training data used for cross-validation.
    method:
        ``"grid"`` for exhaustive search or ``"random"`` for random sampling.
    n_iter:
        Number of random samples evaluated when ``method='random'``.
    cv:
        Number of stratified cross-validation folds.
    scoring:
        Metric understood by :func:`sklearn.model_selection.cross_val_score`.
    random_state:
        Optional random seed for reproducibility.

    Returns
    -------
    best_params, results:
        Dictionary with the best hyperparameters and a dataframe summarizing all
        evaluations sorted by mean cross-validation performance.
    """

    X, y = check_X_y(X, y, accept_sparse=False, dtype=float)
    rng = check_random_state(random_state)

    if method not in {"grid", "random"}:
        raise ValueError("method must be either 'grid' or 'random'.")

    if method == "grid":
        param_iterable = list(ParameterGrid(param_grid))
    else:
        param_iterable = list(
            ParameterSampler(param_grid, n_iter=n_iter, random_state=rng)
        )

    results: List[Dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rng)
    for params in param_iterable:
        model = estimator_cls(**params)
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)
        results.append({
            **params,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mean_score", ascending=False)
    key_list = list(param_grid.keys())
    best_params = results_df.iloc[0][key_list].to_dict()
    return best_params, results_df.reset_index(drop=True)


__all__ = [
    "BaseOrdinalRegressor",
    "CumulativeOrdinalRegressor",
    "SequentialOrdinalRegressor",
    "AdjacentCategoryOrdinalRegressor",
    "load_ordinal_excel",
    "evaluate_models",
    "plot_fragility_curves",
    "hyperparameter_search",
]
