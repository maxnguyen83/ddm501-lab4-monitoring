"""
SHAP explanations for individual scoring decisions.

"""

import logging
from typing import Any, Dict, List

import numpy as np

from pipeline.preprocessing import add_derived_features

logger = logging.getLogger(__name__)


class Explainer:
    """Wraps a SHAP TreeExplainer around the fitted pipeline.

    The pipeline is (features -> classifier). SHAP needs the raw estimator and
    the TRANSFORMED matrix, so this class holds both halves and does the
    transformation itself. Handing the whole pipeline to shap and hoping is
    the usual mistake.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.feature_step = pipeline.named_steps["features"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = list(
            self.feature_step.named_steps["preprocess"].get_feature_names_out()
        )
        self._explainer = None

    def _ensure_explainer(self):
        """Build the explainer on first use.

        Lazily, because importing shap costs about a second and a container
        that is not asked for explanations should not pay it at startup.
        """
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.classifier)
        return self._explainer

    def explain(self, frame, top_n: int = 8) -> Dict[str, Any]:
        """Return the features that moved this one score, largest first.

        TASK 10:
          - Transform the frame with self.feature_step, then call
            shap_values on self.classifier's explainer.
          - Depending on the model, shap returns (n, features) or
            (n, features, classes). Normalise to the positive class.
          - expected_value may be a scalar or an array; normalise it too.
          - Sort by ABSOLUTE contribution and keep the top n.
          - Report the applicant's OWN value for each feature, not the
            standardised one. add_derived_features(frame) gives you the
            derived ones; one-hot columns have no counterpart in the
            application and fall back to the transformed value. An adverse
            action notice quoting "your PAY_0 was 2.16" cannot be reconciled
            with the application form by anyone outside the ML team.
        """
        explainer = self._ensure_explainer()
        transformed = np.asarray(self.feature_step.transform(frame), dtype=float)
        contributions = self._positive_class(explainer.shap_values(transformed))[0]
        base_value = self._base_value(explainer.expected_value)

        own_values = self._own_values(frame)
        order = np.argsort(-np.abs(contributions))[:top_n]

        factors: List[Dict[str, Any]] = []
        for i in order:
            name = self.feature_names[i]
            contribution = float(contributions[i])
            value = own_values.get(name)
            if value is None:
                # One-hot columns (SEX_2, EDUCATION_1, ...) have no single
                # counterpart on the form; the 0/1 indicator is the honest value.
                value = float(transformed[0, i])
            factors.append({
                "feature": name,
                "value": value,
                "contribution": contribution,
                "direction": "increases risk" if contribution > 0 else "reduces risk",
            })

        return {"base_value": base_value, "contributions": factors}

    # -------------------------------------------------------------------------
    @staticmethod
    def _positive_class(shap_values: Any) -> np.ndarray:
        """SHAP values for the default class, shape (n, features).

        Older shap returns a list with one array per class, newer ones a 3-D
        array for multi-output models, and gradient boosting a single 2-D
        array of log-odds. All three are normalised here.
        """
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[..., -1]
        return values

    @staticmethod
    def _base_value(expected_value: Any) -> float:
        """expected_value is a scalar for one output, an array for several."""
        flat = np.asarray(expected_value, dtype=float).ravel()
        return float(flat[-1])

    def _own_values(self, frame) -> Dict[str, float]:
        """The applicant's values in the units of the application form.

        A ratio that is undefined for this applicant (no bill, so no payment
        ratio) was median-imputed inside the pipeline; the imputed value is
        what the model actually used, so that is what is reported.
        """
        derived = add_derived_features(frame).iloc[0]
        medians = self._imputed_medians()
        values: Dict[str, float] = {}
        for name in self.feature_names:
            if name not in derived.index:
                continue
            value = float(derived[name])
            if np.isnan(value):
                if name not in medians:
                    continue
                value = medians[name]
            values[name] = value
        return values

    def _imputed_medians(self) -> Dict[str, float]:
        """Training medians from the numeric imputer, by column name."""
        try:
            preprocess = self.feature_step.named_steps["preprocess"]
            numeric = preprocess.named_transformers_["num"]
            columns = next(cols for name, _, cols in preprocess.transformers_ if name == "num")
            statistics = numeric.named_steps["impute"].statistics_
            return {c: float(s) for c, s in zip(columns, statistics)}
        except (KeyError, StopIteration, AttributeError):  # pragma: no cover
            logger.warning("No imputer found; undefined ratios fall back to scaled values")
            return {}
