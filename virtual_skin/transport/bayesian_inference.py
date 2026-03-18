"""
Hierarchical Bayesian calibration for transport parameters.

Separates: global drug effect, donor effect, site effect, tissue-state effect.
Prevents donor noise from being mis-learned as tissue mechanism.

Uses Pyro for MCMC / variational inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
    from pyro.optim import ClippedAdam
    HAS_PYRO = True
except ImportError:
    HAS_PYRO = False


PARAM_PRIORS = {
    "log_D_sc": (-5.0, 1.5),
    "log_K_sc_ve": (1.0, 0.8),
    "log_D_ve": (-3.5, 1.0),
    "log_k_bind": (-2.5, 1.0),
    "log_k_clear": (-2.0, 0.8),
}


class HierarchicalBayesianCalibrator:
    """Fit transport parameters with hierarchical priors over donors and conditions.

    Generative model:
      θ_global ~ Normal(µ_prior, σ_prior)           [drug-level]
      θ_donor  ~ Normal(θ_global, σ_donor)           [donor random effect]
      θ_site   ~ Normal(θ_donor, σ_site)             [site random effect]
      θ_obs    = θ_site + f(tissue_state, drug_desc) [state modulation]
      y_obs    ~ Normal(forward_model(θ_obs), σ_obs) [IVPT endpoints]
    """

    def __init__(
        self,
        forward_model_fn: Any = None,
        num_samples: int = 2000,
        warmup_steps: int = 1000,
        num_chains: int = 4,
    ) -> None:
        if not HAS_PYRO:
            raise ImportError("Pyro is required for Bayesian calibration: pip install pyro-ppl")
        self.forward_fn = forward_model_fn
        self.num_samples = num_samples
        self.warmup = warmup_steps
        self.num_chains = num_chains
        self.posterior_samples: Optional[Dict] = None

    def _model(
        self,
        observations: Dict[str, torch.Tensor],
        donor_ids: torch.Tensor,
        site_ids: torch.Tensor,
        state_vecs: torch.Tensor,
        drug_vecs: torch.Tensor,
    ) -> None:
        n_donors = int(donor_ids.max().item()) + 1
        n_sites = int(site_ids.max().item()) + 1
        n_obs = donor_ids.shape[0]

        # Global parameter priors
        global_params = {}
        for name, (mu, sigma) in PARAM_PRIORS.items():
            global_params[name] = pyro.sample(
                f"global_{name}", dist.Normal(mu, sigma)
            )

        # Donor-level random effects
        sigma_donor = pyro.sample("sigma_donor", dist.HalfNormal(0.5))
        with pyro.plate("donors", n_donors):
            donor_offsets = {}
            for name in PARAM_PRIORS:
                donor_offsets[name] = pyro.sample(
                    f"donor_{name}", dist.Normal(0, sigma_donor)
                )

        # Site-level random effects
        sigma_site = pyro.sample("sigma_site", dist.HalfNormal(0.3))
        with pyro.plate("sites", n_sites):
            site_offsets = {}
            for name in PARAM_PRIORS:
                site_offsets[name] = pyro.sample(
                    f"site_{name}", dist.Normal(0, sigma_site)
                )

        # Observation noise
        sigma_obs = pyro.sample("sigma_obs", dist.HalfNormal(0.5))

        # State-modulation effect (simple linear for identifiability)
        state_coefs = {}
        for name in PARAM_PRIORS:
            n_state = state_vecs.shape[1]
            state_coefs[name] = pyro.sample(
                f"beta_{name}", dist.Normal(torch.zeros(n_state), 0.2 * torch.ones(n_state))
            )

        with pyro.plate("obs", n_obs):
            for name in PARAM_PRIORS:
                theta = (
                    global_params[name]
                    + donor_offsets[name][donor_ids]
                    + site_offsets[name][site_ids]
                    + (state_vecs * state_coefs[name]).sum(dim=-1)
                )
                # Use the combined theta as the log10-parameter
                if name == "log_D_sc":
                    obs_key = "Jss"
                elif name == "log_K_sc_ve":
                    obs_key = "lag_time"
                else:
                    obs_key = None

                if obs_key and obs_key in observations:
                    pyro.sample(
                        f"y_{obs_key}",
                        dist.Normal(theta, sigma_obs),
                        obs=observations[obs_key],
                    )

    def fit(
        self,
        observations: Dict[str, np.ndarray],
        donor_ids: np.ndarray,
        site_ids: np.ndarray,
        state_vecs: np.ndarray,
        drug_vecs: np.ndarray,
        method: str = "nuts",
    ) -> Dict[str, np.ndarray]:
        """Run Bayesian inference."""
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in observations.items()}
        d_ids = torch.tensor(donor_ids, dtype=torch.long)
        s_ids = torch.tensor(site_ids, dtype=torch.long)
        sv = torch.tensor(state_vecs, dtype=torch.float32)
        dv = torch.tensor(drug_vecs, dtype=torch.float32)

        if method == "nuts":
            kernel = NUTS(self._model)
            mcmc = MCMC(kernel, num_samples=self.num_samples, warmup_steps=self.warmup,
                        num_chains=self.num_chains)
            mcmc.run(obs_t, d_ids, s_ids, sv, dv)
            self.posterior_samples = {k: v.numpy() for k, v in mcmc.get_samples().items()}
        else:
            guide = pyro.infer.autoguide.AutoNormal(self._model)
            optim = ClippedAdam({"lr": 0.01})
            svi = SVI(self._model, guide, optim, loss=Trace_ELBO())
            for _ in range(self.num_samples):
                svi.step(obs_t, d_ids, s_ids, sv, dv)
            self.posterior_samples = {
                k: guide.median()[k].detach().numpy()
                for k in guide.median()
            }

        return self.posterior_samples

    def posterior_summary(self) -> Dict[str, Dict[str, float]]:
        """Return mean and 90% CI for each global parameter."""
        if self.posterior_samples is None:
            raise RuntimeError("Call fit() first.")
        summary = {}
        for k, v in self.posterior_samples.items():
            if k.startswith("global_"):
                summary[k] = {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "ci_5": float(np.percentile(v, 5)),
                    "ci_95": float(np.percentile(v, 95)),
                }
        return summary

    def posterior_predictive_params(
        self, donor_id: int, site_id: int, state_vec: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Given a specific donor/site/state, return (mean, std) for each transport param."""
        if self.posterior_samples is None:
            raise RuntimeError("Call fit() first.")
        result = {}
        for name in PARAM_PRIORS:
            g = self.posterior_samples.get(f"global_{name}", np.array([PARAM_PRIORS[name][0]]))
            d = self.posterior_samples.get(f"donor_{name}", np.zeros_like(g))
            s = self.posterior_samples.get(f"site_{name}", np.zeros_like(g))

            d_eff = d[:, donor_id] if d.ndim > 1 and d.shape[1] > donor_id else d.flatten()
            s_eff = s[:, site_id] if s.ndim > 1 and s.shape[1] > site_id else s.flatten()

            beta = self.posterior_samples.get(f"beta_{name}", np.zeros(state_vec.shape[0]))
            if beta.ndim > 1:
                state_eff = (beta @ state_vec)
            else:
                state_eff = np.dot(beta, state_vec)

            theta = g.flatten() + d_eff + s_eff + state_eff
            result[name] = (float(np.mean(theta)), float(np.std(theta)))
        return result
