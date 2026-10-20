"""Acquisition methods for the ε-MOBO KNO1 example.

The GP and acquisition code uses maximization internally. ``run_demo.py``
negates the deterministic minimization objectives once before fitting and
retains their original values for reporting hypervolume. Batch selection
supports HVI, sequential HVC, and decision-space maximin.

Requires BoTorch, GPyTorch, PyTorch, NumPy, pymoo, and SciPy.
"""

import numpy as np
import torch
from enum import Enum
from typing import List, Optional

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

DTYPE = torch.double

# Version tag embedded in every result file for provenance checking.
# Bump whenever algorithmic behavior changes.
CODE_VERSION = "batch-q1248-v4.7-kno1-demo-sequential-hvc-2026-10-20"
GP_FIT_POLICY = "singletaskgp-defaults-except-matern52-10-attempts-best-per-objective-flexible-nobj"


def log(msg: str) -> None:
    print(msg, flush=True)


# PROJECT-SPECIFIC NUMERICAL SAFEGUARD (not a paper-derived acquisition rule).
# During the completed q=4 debugging, the default torch.cdist matrix-
# multiplication path rounded some very small distances to zero, causing false
# duplicate/observed-point failures. We therefore use the direct Euclidean path
# for every q in {1,2,4,8}.
# This is a numerical safeguard applied to every batch size.
DIRECT_CDIST_MODE = "donot_use_mm_for_euclid_dist"
NOVELTY_TOL = 1e-8
BATCH_COMPLETION_POLICY = (
    "selected-then-nsga-population-then-deterministic-sobol"
)
RESEARCH_SCOPE_NOTE = (
    "Paper-derived method components plus explicitly labelled project-specific "
    "budget, novelty, completion, and fail-fast safeguards."
)


def _direct_cdist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Numerically stable Euclidean distances for novelty checks."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("distance inputs must be 2-D tensors")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("distance inputs have different dimensions")
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.empty(
            (X.shape[0], Y.shape[0]), dtype=X.dtype, device=X.device
        )
    return torch.cdist(
        X, Y.to(dtype=X.dtype, device=X.device), p=2.0,
        compute_mode=DIRECT_CDIST_MODE,
    )


def _validate_novel_batch(
    X_batch: torch.Tensor,
    observed_x: Optional[torch.Tensor],
    q: int,
    d: int,
    tol: float = NOVELTY_TOL,
) -> torch.Tensor:
    """Validate shape, bounds, within-batch uniqueness, and novelty."""
    X_batch = X_batch.reshape(-1, d)
    if tuple(X_batch.shape) != (q, d):
        raise RuntimeError(
            f"candidate batch shape {tuple(X_batch.shape)} != {(q, d)}"
        )
    if not torch.isfinite(X_batch).all():
        raise RuntimeError("candidate batch contains NaN/Inf")
    if ((X_batch < 0.0) | (X_batch > 1.0)).any():
        raise RuntimeError("candidate batch is outside normalized [0,1]^d")

    if q > 1:
        D = _direct_cdist(X_batch, X_batch)
        mask = ~torch.eye(q, dtype=torch.bool, device=X_batch.device)
        if torch.any(D[mask] <= tol):
            raise RuntimeError("batch completion produced duplicate points")

    if observed_x is not None and observed_x.shape[0] > 0:
        observed = observed_x.to(dtype=X_batch.dtype, device=X_batch.device)
        if torch.any(_direct_cdist(X_batch, observed) <= tol):
            raise RuntimeError(
                "batch completion reproduced an observed decision point"
            )
    return X_batch


def _complete_batch_from_population(
    selected_x: torch.Tensor,
    population_x: Optional[torch.Tensor],
    observed_x: Optional[torch.Tensor],
    q: int,
    d: Optional[int] = None,
    tol: float = NOVELTY_TOL,
    sobol_seed: Optional[int] = None,
) -> torch.Tensor:
    """Construct exactly ``q`` unique, unobserved candidates.

    Candidate priority is: (1) acquisition-selected points, (2) remaining
    points from the same generated NSGA-II population, and only when those
    are insufficient, (3) deterministic scrambled-Sobol augmentation.

    Research boundary
    -----------------
    NSGA-II is the published evolutionary search engine [deb2002nsga2], and
    Sobol sequences are standard low-discrepancy designs [joe2008constructing].
    The *ordering and completion policy itself* is a project-specific safeguard
    introduced after the q=4 debugging. Sobol is used only to repair a short or
    duplicate batch after a valid acquisition run; it never replaces failed GP
    fitting or failed acquisition optimization.
    """
    if q <= 0:
        raise ValueError("q must be positive")
    if selected_x.ndim != 2:
        raise ValueError("selected_x must be a 2-D tensor")
    if selected_x.shape[0] == 0 and (
        population_x is None or population_x.shape[0] == 0
    ):
        raise RuntimeError("acquisition returned no candidate points")
    d = int(selected_x.shape[1] if d is None else d)
    if selected_x.shape[1] != d:
        raise ValueError("selected_x dimension mismatch")
    dtype, device = selected_x.dtype, selected_x.device

    if not torch.isfinite(selected_x).all():
        raise RuntimeError("acquisition-selected candidates contain NaN/Inf")
    if ((selected_x < 0.0) | (selected_x > 1.0)).any():
        raise RuntimeError("acquisition-selected candidates are outside [0,1]^d")

    observed = None
    if observed_x is not None:
        if observed_x.ndim != 2 or observed_x.shape[1] != d:
            raise ValueError("observed_x has incompatible shape")
        observed = observed_x.to(dtype=dtype, device=device)
        if not torch.isfinite(observed).all():
            raise RuntimeError("observed_x contains NaN/Inf")

    chosen: list[torch.Tensor] = []

    def add_pool(pool: Optional[torch.Tensor]) -> None:
        if pool is None or len(chosen) >= q:
            return
        if pool.ndim != 2 or pool.shape[1] != d:
            raise ValueError("candidate population has incompatible shape")
        pool = pool.to(dtype=dtype, device=device)
        valid = torch.isfinite(pool).all(dim=1)
        valid &= ((pool >= 0.0) & (pool <= 1.0)).all(dim=1)
        for point in pool[valid]:
            if len(chosen) >= q:
                break
            p = point.reshape(1, d)
            if observed is not None and observed.shape[0] > 0:
                if torch.any(_direct_cdist(p, observed) <= tol):
                    continue
            if chosen:
                current = torch.stack(chosen, dim=0)
                if torch.any(_direct_cdist(p, current) <= tol):
                    continue
            chosen.append(point.clone())

    # Keep the acquisition's ordering first, then use the generated population.
    add_pool(selected_x)
    add_pool(population_x)

    if len(chosen) < q:
        if sobol_seed is None:
            # Deterministic under the experiment's paired torch seed.
            sobol_seed = int(
                torch.randint(0, 2**31 - 1, (1,), device="cpu").item()
            )
        engine = torch.quasirandom.SobolEngine(
            dimension=d, scramble=True, seed=int(sobol_seed)
        )
        # Draw in moderate blocks so duplicate filtering remains inexpensive.
        for _ in range(64):
            needed = q - len(chosen)
            draw_n = max(64, 16 * needed)
            sobol = engine.draw(draw_n, dtype=dtype).to(device=device)
            add_pool(sobol)
            if len(chosen) >= q:
                break

    if len(chosen) < q:
        raise RuntimeError(
            f"candidate completion could not construct q={q} unique, "
            "unobserved points even after deterministic Sobol augmentation"
        )

    result = torch.stack(chosen[:q], dim=0)
    return _validate_novel_batch(result, observed, q, d, tol)


def complete_candidate_batch(
    selected_x: torch.Tensor,
    observed_x: Optional[torch.Tensor],
    q: int,
    population_x: Optional[torch.Tensor] = None,
    tol: float = NOVELTY_TOL,
) -> torch.Tensor:
    """Public q=1,2,4,8 wrapper for the project-specific completion safeguard."""
    return _complete_batch_from_population(
        selected_x=selected_x, population_x=population_x,
        observed_x=observed_x, q=q, d=selected_x.shape[-1], tol=tol,
    )


def validate_candidate_batch(
    X_batch: torch.Tensor, observed_x: Optional[torch.Tensor], q: int, d: int
) -> torch.Tensor:
    """Validate the project batch contract; this is not an acquisition paper claim."""
    return _validate_novel_batch(X_batch, observed_x, q, d)


# ============================================================
# SELECTION CRITERIA
# ============================================================
class SelectionCriteria(Enum):
    # DS-Maximin: greedy farthest-point (maximin) selection in decision
    #   space, following qPOTS [renganathan2025qpots].
    DS_MAXIMIN   = "ds_maximin"
    # HVC: exclusive contribution within the generated candidate front.
    #   Used here as a controlled selection-rule case study; this is distinct
    #   from TSEMO's original hypervolume IMPROVEMENT rule.
    HVC          = "hvc"
    # HVI: sequential hypervolume improvement relative to the already
    #   observed objective set. This is the original TSEMO selection rule.
    HVI          = "hvi"
    # Tchebycheff: weighted Tchebycheff scalarization, as in MOEA/D
    #   [zhang2007moead] and ParEGO [knowles2006parego].
    TCHEBYCHEFF  = "tchebycheff"


# ============================================================
# SHARED SELECTION FUNCTIONS
# ============================================================
def _normalize(Z: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each column to [0,1]."""
    lo = Z.min(dim=0).values
    hi = Z.max(dim=0).values
    scale = (hi - lo).clamp(min=1e-8)
    return (Z - lo) / scale


def _maximin_from_observed(
    X_candidates: torch.Tensor,
    X_observed: torch.Tensor,
    q: int,
) -> torch.Tensor:
    """Sequential qPOTS maximin selection with direct distances.

    qPOTS chooses points on a posterior-sampled Pareto set using a maximin
    decision-space criterion [renganathan2025qpots]. Here each accepted point
    is appended to the reference set so later batch members are separated from
    observed designs and earlier members. The direct-distance tolerance and
    short-front completion are project safeguards, not qPOTS paper claims.
    """
    if X_candidates.ndim != 2 or X_observed.ndim != 2:
        raise ValueError("maximin inputs must be 2-D tensors")
    if X_candidates.shape[0] == 0:
        raise RuntimeError("maximin received an empty candidate set")
    if X_observed.shape[0] == 0:
        raise RuntimeError("qPOTS maximin requires previously observed points")
    if X_candidates.shape[1] != X_observed.shape[1]:
        raise ValueError("candidate/observed decision dimensions do not match")

    reference = X_observed.to(
        dtype=X_candidates.dtype, device=X_candidates.device
    )
    available = torch.ones(
        X_candidates.shape[0], dtype=torch.bool, device=X_candidates.device
    )
    selected: list[int] = []
    for _ in range(min(q, X_candidates.shape[0])):
        min_dist = _direct_cdist(X_candidates, reference).min(dim=1).values
        min_dist[~available] = -float("inf")
        # Do not select a point that reproduces an observed or already selected
        # design. Completion from the NSGA-II population handles short fronts.
        min_dist[min_dist <= NOVELTY_TOL] = -float("inf")
        idx = int(min_dist.argmax())
        if not torch.isfinite(min_dist[idx]):
            break
        selected.append(idx)
        available[idx] = False
        reference = torch.cat([reference, X_candidates[idx:idx + 1]], dim=0)
    return torch.tensor(selected, dtype=torch.long, device=X_candidates.device)


def _hvc_selection(Y: torch.Tensor, q: int,
                   ref_point: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Select one point at a time, recomputing exclusive HVC after removal.

    Normalize once over the original candidate set, as specified in the
    manuscript. The remaining nondominated set changes after each pick.
    References: [zitzler1999hypervolume, beume2007smsemoa].
    """
    M = Y.shape[0]
    if M == 0 or q <= 0:
        return torch.empty(0, dtype=torch.long, device=Y.device)

    # Scale-invariant HVC: normalize the candidate front to [0,1]^K first.
    # HVC depends on scale: objectives with larger numeric ranges can
    # dominate the contributions. Normalize the candidate front first.
    Y_min = Y.min(dim=0).values
    Y_rng = (Y.max(dim=0).values - Y_min).clamp(min=1e-12)
    Y = (Y - Y_min) / Y_rng
    if ref_point is None:
        # MAXIMIZATION convention (BoTorch): ref must be dominated by all
        # points, i.e. BELOW the (now normalized) front.
        ref_point = -0.1 * torch.ones(Y.shape[1], dtype=Y.dtype)
    else:
        ref_point = (ref_point - Y_min) / Y_rng

    remaining = torch.where(is_non_dominated(Y))[0].tolist()
    selected: list[int] = []
    for _ in range(min(q, len(remaining))):
        current = Y[remaining]
        hv_total = _hypervolume_max(current, ref_point)
        contributions = [
            hv_total - _hypervolume_max(
                current[torch.arange(len(remaining), device=Y.device) != pos],
                ref_point,
            )
            for pos in range(len(remaining))
        ]
        best_position = max(range(len(remaining)), key=contributions.__getitem__)
        selected.append(remaining.pop(best_position))
    return torch.tensor(selected, dtype=torch.long, device=Y.device)


def _hypervolume_max(Y: torch.Tensor, ref_point: torch.Tensor) -> float:
    """Hypervolume for MAXIMIZATION data, ignoring points that do not
    dominate the reference point."""
    if Y.numel() == 0:
        return 0.0
    valid = torch.all(Y >= ref_point, dim=1)
    Yv = Y[valid]
    if Yv.shape[0] == 0:
        return 0.0
    Yv = Yv[is_non_dominated(Yv)]
    return float(Hypervolume(ref_point).compute(Yv))


def _hvi_selection(
    Y_candidates: torch.Tensor,
    Y_observed: torch.Tensor,
    q: int,
    ref_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sequential hypervolume-improvement selection (TSEMO rule).

    All values use the MAXIMIZATION convention and must be on the same scale.
    When no reference is supplied, mirror the original TSEMO construction:
    use the sampled-front nadir with a 1% outward offset.
    """
    if Y_candidates.ndim != 2 or Y_observed.ndim != 2:
        raise ValueError("HVI inputs must be 2-D tensors")
    if Y_candidates.shape[0] == 0:
        raise RuntimeError("HVI received an empty candidate set")
    if Y_candidates.shape[1] != Y_observed.shape[1]:
        raise ValueError("candidate/observed objective dimensions do not match")
    if not torch.isfinite(Y_candidates).all() or not torch.isfinite(Y_observed).all():
        raise RuntimeError("HVI received NaN/Inf objective values")

    if ref_point is None:
        y_min = Y_candidates.min(dim=0).values
        y_rng = (Y_candidates.max(dim=0).values - y_min).clamp(min=1e-6)
        ref_point = y_min - 0.01 * y_rng
    else:
        ref_point = ref_point.to(dtype=Y_candidates.dtype, device=Y_candidates.device)

    augmented = Y_observed.to(dtype=Y_candidates.dtype, device=Y_candidates.device).clone()
    selected = []
    n_select = min(q, Y_candidates.shape[0])
    for _ in range(n_select):
        hv_base = _hypervolume_max(augmented, ref_point)
        improvements = torch.full(
            (Y_candidates.shape[0],), -float("inf"),
            dtype=Y_candidates.dtype, device=Y_candidates.device,
        )
        for i in range(Y_candidates.shape[0]):
            if i in selected:
                continue
            hv_new = _hypervolume_max(
                torch.cat([augmented, Y_candidates[i:i + 1]], dim=0),
                ref_point,
            )
            improvements[i] = hv_new - hv_base
        idx = int(improvements.argmax())
        selected.append(idx)
        augmented = torch.cat([augmented, Y_candidates[idx:idx + 1]], dim=0)

    return torch.tensor(selected, dtype=torch.long, device=Y_candidates.device)


def _tchebycheff_selection(Y: torch.Tensor, q: int) -> torch.Tensor:
    """Select via random-weight Tchebycheff scalarization.
    References: MOEA/D [zhang2007moead]; ParEGO [knowles2006parego]."""
    M, K = Y.shape
    if M <= q:
        return torch.arange(M, dtype=torch.long)

    ideal = Y.max(dim=0).values   # ideal point under MAXIMIZATION convention
    weights = torch.distributions.Dirichlet(torch.ones(K, dtype=Y.dtype)).sample((q,))
    selected = []

    for j in range(q):
        diff = torch.abs(Y - ideal.unsqueeze(0))
        tcheby = (weights[j].unsqueeze(0) * diff).max(dim=1).values
        for s in selected:
            tcheby[s] = float("inf")
        selected.append(int(tcheby.argmin()))

    return torch.tensor(selected, dtype=torch.long)


def select_from_pareto_front(
    X_pareto: torch.Tensor,
    Y_pareto: torch.Tensor,
    q: int,
    criteria: SelectionCriteria,
    ref_point: Optional[torch.Tensor] = None,
    observed_x: Optional[torch.Tensor] = None,
    observed_y: Optional[torch.Tensor] = None,
    population_x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select and complete a general-q batch from a generated front.

    The configured criterion determines the priority ordering:

    * DS-Maximin follows the diversity principle of qPOTS
      [renganathan2025qpots].
    * Sequential HVI follows TSEMO [bradford2018tsemo].
    * HVC is a controlled project case study based on the hypervolume indicator
      [zitzler1999hypervolume, beume2007smsemoa].

    If the selected front is too short or contains near-duplicates, remaining
    points are taken from the same NSGA-II population and then, only as
    project-specific duplicate repair, from a deterministic Sobol sequence.
    """
    M, d = X_pareto.shape
    if M == 0:
        raise RuntimeError("select_from_pareto_front received an empty front")

    if criteria == SelectionCriteria.DS_MAXIMIN:
        if observed_x is None:
            raise ValueError("DS_MAXIMIN requires observed_x")
        indices = _maximin_from_observed(X_pareto, observed_x, q)
    elif criteria == SelectionCriteria.HVC:
        indices = _hvc_selection(Y_pareto, q, ref_point)
    elif criteria == SelectionCriteria.HVI:
        if observed_y is None:
            raise ValueError("HVI requires observed_y")
        indices = _hvi_selection(Y_pareto, observed_y, q, ref_point)
    elif criteria == SelectionCriteria.TCHEBYCHEFF:
        indices = _tchebycheff_selection(Y_pareto, q)
    else:
        raise ValueError(f"Unknown selection criterion: {criteria}")

    initial = X_pareto[indices[:q]] if indices.numel() else X_pareto[:0]
    return _complete_batch_from_population(
        selected_x=initial,
        population_x=population_x if population_x is not None else X_pareto,
        observed_x=observed_x,
        q=q,
        d=d,
    )


# ============================================================
# SPECTRAL SAMPLER (for EpsilonTSEMO)
# ============================================================
class SpectralSampler:
    """
    Random Fourier Feature approximation to a GP posterior.

    Given GP hyperparameters (lengthscale, outputscale, noise),
    builds a Bayesian linear model in feature space:
        f(x) ≈ φ(x)ᵀβ,  β ~ N(μ_β, Σ_β)

    KERNEL CONSISTENCY (fix): the spectral weights W are drawn from the
    spectral density of the SAME kernel family as the fitted GP:
      - Matern-ν (BoTorch SingleTaskGP default, ν=5/2): the spectral
        density is a multivariate Student-t with 2ν degrees of freedom,
        sampled via the Gaussian scale-mixture  W = (z/ℓ)/sqrt(u/(2ν)),
        z ~ N(0, I), u ~ χ²(2ν).
      - SE (RBF): W = z/ℓ, z ~ N(0, I)  (Gaussian spectral density).
    Previously W was always drawn from the SE spectral density regardless
    of the GP's actual kernel, so ε-TSEMO's sample paths had the wrong
    smoothness/correlation structure whenever the GP used Matern-5/2.

    `lengthscale` may be a scalar or a per-dimension (ARD) tensor of
    shape (d,); ARD lengthscales from the fitted GP are used directly.

    References: RFF [rahimi2007random]; sparse-spectrum GP weight-space
    posterior [lazaro2010sparse]; spectral TS as used in TSEMO
    [bradford2018tsemo]; Matern/SE spectral densities as tabulated in
    [bradford2018tsemo] and [do2024epsilon].
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 lengthscale, outputscale: float,
                 noise_var: float, n_features: int,
                 kernel_type: str = "matern", nu: float = 2.5):
        d = train_x.shape[1]
        dtype = train_x.dtype

        # Lengthscale: scalar or ARD vector (d,) — broadcast against (Np, d)
        ls = torch.as_tensor(lengthscale, dtype=dtype).clamp(min=1e-3)
        if ls.ndim > 1:
            ls = ls.reshape(-1)
        self.kernel_type = kernel_type
        self.nu = nu

        # Spectral weights from the kernel's spectral density
        z = torch.randn(n_features, d, dtype=dtype)
        if kernel_type == "matern":
            df = 2.0 * nu
            u = torch.distributions.Chi2(torch.tensor(df, dtype=dtype)) \
                     .sample((n_features, 1))
            self.W = (z / ls) / torch.sqrt(u / df)
        elif kernel_type == "se":
            self.W = z / ls
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type!r} "
                             f"(use 'matern' or 'se')")

        self.b = torch.rand(n_features, dtype=dtype) * 2.0 * torch.pi
        self.scale = torch.sqrt(torch.tensor(2.0 * outputscale / n_features, dtype=dtype))

        # Feature matrix for training points
        Phi = self._features(train_x)  # (N, Np)

        # Bayesian linear regression posterior
        A = Phi.T @ Phi + noise_var * torch.eye(n_features, dtype=dtype)
        L = torch.linalg.cholesky(A)
        Ainv = torch.cholesky_inverse(L)

        self.mu_beta = Ainv @ Phi.T @ train_y        # (Np,)
        self.L_sigma = torch.linalg.cholesky(noise_var * Ainv)  # (Np, Np)
        self.Np = n_features

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute φ(x) = √(2σ²/Np) · cos(Wx + b)."""
        return self.scale * torch.cos(x @ self.W.T + self.b.unsqueeze(0))

    def sample_path(self) -> torch.Tensor:
        """Draw one β ~ N(μ_β, Σ_β)."""
        eps = torch.randn(self.Np, dtype=self.mu_beta.dtype)
        return self.mu_beta + self.L_sigma @ eps

    def evaluate(self, beta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate f(x) = φ(x)ᵀβ."""
        return self._features(x) @ beta


# ============================================================
# GP HYPERPARAMETER EXTRACTION
# ============================================================
def _extract_gp_hyperparams(gp) -> tuple:
    """Strictly extract spectral-sampler hyperparameters from a fitted GP.

    Returns (ARD lengthscale, outputscale, noise variance, kernel type, nu).
    Any unexpected model structure raises an error instead of silently
    substituting defaults, because such a fallback would change TSEMO's
    sampled covariance without being recorded.
    """
    from gpytorch.kernels import MaternKernel, RBFKernel

    if not hasattr(gp, "covar_module"):
        raise RuntimeError("GP has no covar_module")
    covar = gp.covar_module
    base = covar.base_kernel if hasattr(covar, "base_kernel") else covar
    if not hasattr(base, "lengthscale"):
        raise RuntimeError("GP kernel has no lengthscale")

    ls = base.lengthscale.detach().reshape(-1).to(dtype=DTYPE)
    if not torch.isfinite(ls).all() or (ls <= 0).any():
        raise RuntimeError(f"invalid GP lengthscale: {ls}")

    if isinstance(base, MaternKernel):
        kernel_type, nu = "matern", float(base.nu)
    elif isinstance(base, RBFKernel):
        kernel_type, nu = "se", float("nan")
    else:
        raise RuntimeError(f"unsupported GP kernel for spectral sampling: {type(base)}")

    # The BoTorch default dimensionality-scaled covariance has no separate
    # outputscale parameter (it is fixed to one after outcome standardization).
    # ScaleKernel-based models are still supported for backward compatibility.
    outputscale = (
        float(covar.outputscale.detach())
        if hasattr(covar, "outputscale")
        else 1.0
    )
    noise_var = float(gp.likelihood.noise.detach().mean())
    if not np.isfinite(outputscale) or outputscale <= 0:
        raise RuntimeError(f"invalid GP outputscale: {outputscale}")
    if not np.isfinite(noise_var) or noise_var <= 0:
        raise RuntimeError(f"invalid GP noise variance: {noise_var}")

    return ls, outputscale, noise_var, kernel_type, nu


# ============================================================
# EPSILON-TSEMO (Spectral Sampling)
# ============================================================
class EpsilonTSEMO:
    """
    ε-TSEMO: Epsilon-greedy TS via spectral sampling (RFF).

    Exploration (r ≤ ε): Single Thompson sample → NSGA-II → select
    Exploitation (r > ε): Average Ns samples → NSGA-II → select

    References:
      - Base method (spectral TS + GA + HV selection): TSEMO
        [bradford2018tsemo]
      - ε-greedy generic-TS / sample-average-TS switching:
        [do2024epsilon]
      - RFF spectral sampling: [rahimi2007random], [lazaro2010sparse]
      - Thompson sampling principle: [russo2014learning]
      - NSGA-II inner optimizer: [deb2002nsga2]
    """

    def __init__(self, train_x, train_y, gp_list, bounds,
                 nobj, q=1, epsilon=0.1, Ns=50, Np=512,
                 ngen=20, npop=100,
                 selection_criteria=SelectionCriteria.DS_MAXIMIN,
                 ref_point=None, **kwargs):
        self.bounds = bounds
        self.nobj = nobj
        self.q = q
        self.epsilon = epsilon
        self.Ns = Ns
        self.ngen = ngen
        self.npop = npop
        self.dtype = bounds.dtype
        self.selection_criteria = selection_criteria
        self.ref_point = ref_point
        self.observed_x = train_x.to(dtype=self.dtype)

        # Build spectral samplers from GP hyperparameters.
        # Kernel family + nu + ARD lengthscales are read from each fitted
        # GP so the RFF spectral density MATCHES the GP's actual kernel
        # (Matern-5/2 by default in fit_gp_models / BoTorch SingleTaskGP).
        self._samplers = []
        kernel_desc = []
        for k in range(nobj):
            gp = gp_list[k] if k < len(gp_list) else None
            if gp is not None:
                ls, os_, nv, ktype, nu = _extract_gp_hyperparams(gp)
            else:
                ls, os_, nv, ktype, nu = 1.0, 1.0, 1e-3, "matern", 2.5
            target_k = (
                gp.train_targets.detach().reshape(-1).to(dtype=self.dtype)
                if gp is not None and hasattr(gp, "train_targets")
                else train_y[:, k].to(dtype=self.dtype)
            )
            self._samplers.append(
                SpectralSampler(train_x.to(dtype=self.dtype),
                                target_k,
                                ls, os_, nv, Np,
                                kernel_type=ktype, nu=nu)
            )
            kernel_desc.append(f"{ktype}" + (f"(nu={nu:g})" if ktype == "matern" else ""))
        self.observed_y = torch.stack(
            [gp_list[k].train_targets.detach().reshape(-1).to(dtype=self.dtype)
             for k in range(nobj)],
            dim=-1,
        )
        log(f"[ε-TSEMO] d={bounds.shape[1]}, ε={epsilon:.2f}, "
            f"criteria={selection_criteria.value}, Ns={Ns}, Np={Np}, "
            f"kernels={kernel_desc}")

    def _draw_betas(self, n_draws=1):
        return [[s.sample_path() for s in self._samplers] for _ in range(n_draws)]

    def suggest(self) -> torch.Tensor:
        r = float(torch.rand(1))

        if r <= self.epsilon:
            # EXPLORATION: single TS draw
            betas = self._draw_betas(1)[0]
        else:
            # EXPLOITATION: average Ns draws
            all_betas = self._draw_betas(self.Ns)
            betas = [
                torch.stack([all_betas[s][k] for s in range(self.Ns)]).mean(dim=0)
                for k in range(self.nobj)
            ]

        # Shared pymoo NSGA-II (identical optimizer for ALL families).
        # Each fn evaluates the FIXED RFF sample path phi_k(x)^T beta_k.
        fns = [
            (lambda s_, b_: (lambda X: s_.evaluate(b_, X)))(s, b)
            for s, b in zip(self._samplers, betas)
        ]
        X_pf, Y_pf, X_pop = pymoo_front_on_fns(
            fns, self.bounds.shape[1], self.nobj, self.npop, self.ngen,
            dtype=self.dtype, return_population=True,
        )

        if X_pf.shape[0] == 0:
            raise RuntimeError("epsilon-TSEMO received an empty Pareto set")

        return select_from_pareto_front(
            X_pf, Y_pf, self.q, self.selection_criteria, self.ref_point,
            observed_x=self.observed_x, observed_y=self.observed_y,
            population_x=X_pop,
        )



# ============================================================
# FIXED THOMPSON SAMPLE PATHS (BoTorch pathwise posterior sampling)
# Reference: Wilson et al. (2020), "Efficiently sampling functions
# from Gaussian process posteriors" [wilson2020efficiently].
# One path per objective is drawn ONCE per BO iteration and held
# fixed throughout the entire NSGA-II run, as Thompson sampling
# requires. (Previously the posterior was re-sampled at every
# NSGA-II _evaluate() call, yielding a stochastic posterior-
# resampling variant rather than true TS.)
# ============================================================
def draw_posterior_paths(model, nobj: int):
    """Return a list of nobj callables f_k(X)->tensor(N,), each a FIXED
    sample path from the corresponding GP posterior."""
    try:
        from botorch.sampling.pathwise import draw_matheron_paths
    except ImportError as e:
        raise ImportError(
            "botorch.sampling.pathwise.draw_matheron_paths is required for "
            "fixed Thompson sample paths (qPOTS family). Upgrade botorch "
            "(pip install -U botorch)."
        ) from e
    models = list(model.models) if hasattr(model, "models") else [model]
    paths = []
    for k in range(nobj):
        p = draw_matheron_paths(models[k], sample_shape=torch.Size([1]))
        paths.append(p)

    def make_fn(path):
        def fn(X):
            with torch.no_grad():
                out = path(X)
            return out.reshape(-1)
        return fn
    return [make_fn(p) for p in paths]


def pymoo_front_on_fns(
    fns, d: int, nobj: int, npop: int, ngen: int, dtype=None,
    return_population: bool = False,
):
    """Shared NSGA-II front finder in MAXIMIZATION convention.

    When ``return_population`` is true, the final NSGA-II population is also
    returned so general-q batch completion can use acquisition-generated
    candidates before deterministic Sobol augmentation.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize

    dtype = dtype or DTYPE

    class FixedPathProblem(Problem):
        def __init__(self):
            super().__init__(n_var=d, n_obj=nobj,
                             xl=np.zeros(d), xu=np.ones(d))

        def _evaluate(self, x, out, *args, **kwargs):
            X = torch.tensor(x, dtype=dtype)
            if X.ndim == 1:
                X = X.unsqueeze(0)
            cols = [fn(X) for fn in fns]
            F = torch.stack(cols, dim=-1)
            out["F"] = -F.detach().cpu().numpy()

    try:
        res = pymoo_minimize(
            FixedPathProblem(), NSGA2(pop_size=npop),
            termination=("n_gen", ngen),
            seed=np.random.randint(0, 2**31), verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"NSGA-II failed: {exc}") from exc

    if res.X is None or len(res.X) == 0:
        raise RuntimeError("NSGA-II returned an empty Pareto set")

    X_pf = torch.tensor(res.X, dtype=dtype)
    Y_pf = torch.tensor(-res.F, dtype=dtype)
    if X_pf.ndim == 1:
        X_pf = X_pf.unsqueeze(0)
        Y_pf = Y_pf.unsqueeze(0)
    if not torch.isfinite(X_pf).all() or not torch.isfinite(Y_pf).all():
        raise RuntimeError("NSGA-II returned NaN/Inf values")

    X_pop = X_pf
    if getattr(res, "pop", None) is not None:
        raw_pop = res.pop.get("X")
        if raw_pop is not None and len(raw_pop) > 0:
            X_pop = torch.tensor(raw_pop, dtype=dtype)
            if X_pop.ndim == 1:
                X_pop = X_pop.unsqueeze(0)
            if not torch.isfinite(X_pop).all():
                raise RuntimeError("NSGA-II population contains NaN/Inf")

    if return_population:
        return X_pf, Y_pf, X_pop
    return X_pf, Y_pf


# ============================================================
# EPSILON-QPOTS (Reparametrization Trick)
# ============================================================
class EpsilonQPOTS:
    """
    ε-qPOTS: Epsilon-greedy TS via reparametrization trick.

    Exploration (r ≤ ε): Random point in [0,1]^d
    Exploitation (r > ε): ONE fixed posterior sample path per objective → pymoo NSGA-II → select

    References:
      - Base method (exact-posterior TS + NSGA-II + maximin batch):
        qPOTS [renganathan2025qpots]
      - ε-greedy exploration-exploitation switching: [do2024epsilon]
      - Thompson sampling principle: [russo2014learning]
      - GP models: BoTorch [balandat2020botorch]; fixed pathwise sampling: [wilson2020efficiently]
      - NSGA-II inner optimizer: [deb2002nsga2]
    """

    def __init__(self, model_obj, bounds, nobj, q=1, epsilon=0.1,
                 ngen=20, npop=100,
                 selection_criteria=SelectionCriteria.DS_MAXIMIN,
                 ref_point=None, observed_x=None, observed_y=None, **kwargs):
        self.model = model_obj   # ModelListGP (has .posterior())
        self.bounds = bounds
        self.nobj = nobj
        self.q = q
        self.d = bounds.shape[1]
        self.epsilon = epsilon
        self.ngen = ngen
        self.npop = npop
        self.dtype = bounds.dtype
        self.selection_criteria = selection_criteria
        self.ref_point = ref_point
        self.observed_x = observed_x
        self.observed_y = observed_y

        log(f"[ε-qPOTS] d={self.d}, ε={epsilon:.2f}, criteria={selection_criteria.value}")

    def suggest(self) -> torch.Tensor:
        r = float(torch.rand(1))

        # EXPLORATION: random point
        if r <= self.epsilon:
            log(f"  [ε-qPOTS] EXPLORE (r={r:.3f})")
            return torch.rand(self.q, self.d, dtype=self.dtype)

        # EXPLOITATION: reparam-trick TS + NSGA-II
        log(f"  [ε-qPOTS] EXPLOIT (r={r:.3f})")
        return self._exploit_nsga2()

    def _exploit_nsga2(self) -> torch.Tensor:
        """Run NSGA-II on ONE fixed Thompson sample path per objective
        (pathwise posterior sampling; held constant across all NSGA-II
        generations, as Thompson sampling requires)."""
        X_pf, Y_pf, X_pop = self._run_fixed_path_front()
        if X_pf.shape[0] == 0:
            raise RuntimeError("epsilon-qPOTS received an empty Pareto set")
        return select_from_pareto_front(
            X_pf, Y_pf, self.q, self.selection_criteria, self.ref_point,
            observed_x=self.observed_x, observed_y=self.observed_y,
            population_x=X_pop,
        )

    def _run_fixed_path_front(self):
        """Draw one fixed posterior path per objective, then find its
        Pareto front with the shared pymoo NSGA-II."""
        paths = draw_posterior_paths(self.model, self.nobj)
        return pymoo_front_on_fns(
            paths, self.d, self.nobj, self.npop, self.ngen,
            dtype=self.dtype, return_population=True,
        )


# ============================================================
# EPSILON-UCB (Upper Confidence Bound)
# ============================================================
class EpsilonUCB:
    """
    ε-UCB: Epsilon-greedy with Upper Confidence Bound exploitation.

    Exploration (r ≤ ε): Random point in [0,1]^d (same as ε-qPOTS)
    Exploitation (r > ε): NSGA-II on UCB = {μ_k(x) + β·σ_k(x)} for k=1..K
                          Then apply selection criteria to the Pareto front.

    References:
      - UCB acquisition (mu + beta*sigma): GP-UCB [srinivas2010gaussian]
      - Two-stage structure (per-objective acquisition surfaces solved
        jointly as a cheap MO problem via NSGA-II, then select from the
        resulting front): USeMO [belakaria2020uncertainty]
      - Related MO-UCB via random scalarizations (different approach:
        scalarize-then-optimize with DIRECT): [paria2020flexible]
      - ε-greedy switching: [do2024epsilon]
      - NSGA-II inner optimizer: [deb2002nsga2]
    """

    def __init__(self, model_obj, bounds, nobj, q=1, epsilon=0.1,
                 beta=2.0, ngen=20, npop=100,
                 selection_criteria=SelectionCriteria.DS_MAXIMIN,
                 ref_point=None, observed_x=None, observed_y=None, **kwargs):
        self.model = model_obj   # ModelListGP
        self.bounds = bounds
        self.nobj = nobj
        self.q = q
        self.d = bounds.shape[1]
        self.epsilon = epsilon
        self.beta = beta
        self.ngen = ngen
        self.npop = npop
        self.dtype = bounds.dtype
        self.selection_criteria = selection_criteria
        self.ref_point = ref_point
        self.observed_x = observed_x
        self.observed_y = observed_y

        log(f"[ε-UCB] d={self.d}, ε={epsilon:.2f}, β={beta:.1f}, "
            f"criteria={selection_criteria.value}")

    def suggest(self) -> torch.Tensor:
        r = float(torch.rand(1))

        if r <= self.epsilon:
            # EXPLORATION: random point (same as ε-qPOTS)
            log(f"  [ε-UCB] EXPLORE (r={r:.3f}, random point)")
            return torch.rand(self.q, self.d, dtype=self.dtype)
        else:
            # EXPLOITATION: NSGA-II on UCB = μ + β·σ
            log(f"  [ε-UCB] EXPLOIT (r={r:.3f}, UCB with β={self.beta})")
            return self._run_nsga2()

    def _ucb_fns(self):
        """Per-objective deterministic UCB callables μ_k + β·σ_k
        (maximization convention), for the shared pymoo NSGA-II helper."""
        model, beta, nobj = self.model, self.beta, self.nobj

        def make(k):
            def f_k(X):
                with torch.no_grad():
                    post = model.models[k].posterior(X)
                    mu = post.mean.reshape(-1)
                    sd = post.variance.reshape(-1).clamp(min=1e-12).sqrt()
                return mu + beta * sd
            return f_k
        return [make(k) for k in range(nobj)]

    def _ucb_front(self):
        """NSGA-II front of the UCB surfaces via the SHARED pymoo helper
        (identical optimizer across all families). Returns (X_pf, Y_pf)."""
        return pymoo_front_on_fns(
            self._ucb_fns(), self.d, self.nobj, self.npop, self.ngen,
            dtype=self.dtype, return_population=True,
        )

    def _run_nsga2(self) -> torch.Tensor:
        """UCB front + greedy selection by the configured criterion."""
        X_pf, Y_pf, X_pop = self._ucb_front()
        if X_pf.shape[0] == 0:
            raise RuntimeError("epsilon-UCB received an empty Pareto set")
        return select_from_pareto_front(
            X_pf, Y_pf, self.q, self.selection_criteria, self.ref_point,
            observed_x=self.observed_x, observed_y=self.observed_y,
            population_x=X_pop,
        )

# ============================================================
# EPSILON-UCB — EXPLORE-FROM-PARETO-SET VARIANT
# ============================================================
class EpsilonUCBExplorePF(EpsilonUCB):
    """
    eps-UCB variant whose EXPLORATION step samples uniformly at random
    from the Pareto set of the UCB surfaces, instead of uniformly from
    the decision space.

    Exploration (u <= eps): NSGA-II on {mu_k + beta*sigma_k} -> pick q
                            members of the UCB front uniformly at random
    Exploitation (u > eps): identical to EpsilonUCB
                            (UCB front -> selection criterion)

    References: same as EpsilonUCB ([srinivas2010gaussian],
    [belakaria2020uncertainty], [deb2002nsga2]). The explore-from-
    Pareto-set variant itself is NEW in this work (exploration-mode
    study); no prior reference.
    """

    def suggest(self) -> torch.Tensor:
        u = float(torch.rand(1))

        if u <= self.epsilon:
            log(f"  [ε-UCB-PFexp] EXPLORE (u={u:.3f}, random from UCB front)")
            X_pf, _, X_pop = self._ucb_front()
            if X_pf.shape[0] == 0:
                raise RuntimeError("epsilon-UCB-PFexp received an empty Pareto set")
            selected = X_pf[torch.randperm(X_pf.shape[0])[: self.q]]
            return _complete_batch_from_population(
                selected, X_pop, self.observed_x, self.q, self.d
            )

        log(f"  [ε-UCB-PFexp] EXPLOIT (u={u:.3f})")
        return self._run_nsga2()


# ============================================================
# EPSILON-QPOTS — EXPLORE-FROM-PARETO-SET VARIANT (NEW)
# ============================================================
class EpsilonQPOTSExplorePF(EpsilonQPOTS):
    """
    eps-qPOTS variant whose EXPLORATION step samples uniformly at random
    from the TS-sampled Pareto set, instead of uniformly from the
    decision space.

    Exploration (u <= eps): reparam-trick TS -> NSGA-II -> pick q members
                            of the resulting Pareto set uniformly at random
    Exploitation (u > eps): identical to EpsilonQPOTS
                            (reparam-trick TS -> NSGA-II -> selection criterion)

    References: same as EpsilonQPOTS ([renganathan2025qpots],
    [do2024epsilon], [deb2002nsga2]). The explore-from-Pareto-set
    variant itself is NEW in this work (advisor-requested exploration-
    mode study); no prior reference.
    """

    def suggest(self) -> torch.Tensor:
        u = float(torch.rand(1))

        if u <= self.epsilon:
            # EXPLORATION: random member(s) of the sampled Pareto set
            log(f"  [eps-qPOTS-PFexp] EXPLORE (u={u:.3f}, random from TS Pareto set)")
            X_pf, _, X_pop = self._run_nsga2_front()
            if X_pf.shape[0] == 0:
                raise RuntimeError("epsilon-qPOTS-PFexp received an empty Pareto set")
            selected = X_pf[torch.randperm(X_pf.shape[0])[: self.q]]
            return _complete_batch_from_population(
                selected, X_pop, self.observed_x, self.q, self.d
            )

        # EXPLOITATION: identical to parent
        log(f"  [eps-qPOTS-PFexp] EXPLOIT (u={u:.3f})")
        return self._exploit_nsga2()

    def _run_nsga2_front(self):
        """Same fixed-path TS + shared NSGA-II machinery as exploitation,
        but returns the raw Pareto front without applying any selection
        criterion. (Delegates to EpsilonQPOTS._run_fixed_path_front.)"""
        return self._run_fixed_path_front()



# ============================================================
# EPSILON-GREEDY MEAN/STD PARETO-FRONT MOBO
# Richardson, De Ath & Chugh (GECCO 2024 Companion):
#   - exploitation: optimize the vector of GP posterior means;
#   - exploration:  optimize the vector of GP posterior standard deviations;
#   - epsilon-greedy: choose the uncertainty front with probability epsilon,
#     otherwise choose the mean front;
#   - select uniformly at random from the chosen Pareto front.
# ============================================================
class EpsilonGreedyMeanStdPF:
    """Direct epsilon-greedy MOBO using separate mean and uncertainty fronts.

    Research boundary: the Bernoulli switching idea is motivated by
    epsilon-greedy Thompson sampling [do2024epsilon], and NSGA-II
    [deb2002nsga2] approximates the mean/std Pareto fronts. The exact
    mean-front versus standard-deviation-front method is project-defined; no
    cited paper is claimed to propose this exact algorithm.

    All GP outputs follow the module-wide MAXIMIZATION convention. Therefore,
    the exploitative front maximizes posterior means of ``g=-f`` and the
    exploratory front maximizes posterior standard deviations. ``policy`` may
    be ``"epsilon"`` (Bernoulli switch), ``"mean"`` (fully exploitative), or
    ``"std"`` (fully exploratory).
    """

    VALID_POLICIES = {"epsilon", "mean", "std"}

    def __init__(self, model_obj, bounds, nobj, q=1, epsilon=0.1,
                 ngen=20, npop=100, policy="epsilon", **kwargs):
        if policy not in self.VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(self.VALID_POLICIES)}; got {policy}"
            )
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError(f"epsilon must be in [0,1]; got {epsilon}")
        self.model = model_obj
        self.bounds = bounds
        self.nobj = int(nobj)
        self.q = int(q)
        self.d = int(bounds.shape[1])
        self.epsilon = float(epsilon)
        self.ngen = int(ngen)
        self.npop = int(npop)
        self.policy = policy
        self.dtype = bounds.dtype
        self.observed_x = kwargs.get("observed_x")
        log(
            f"[epsilon-Greedy-MOBO] d={self.d}, epsilon={self.epsilon:.2f}, "
            f"policy={self.policy}, selection=uniform-from-chosen-PF"
        )

    def _posterior_fns(self, surface: str):
        if surface not in {"mean", "std"}:
            raise ValueError(f"unknown posterior surface: {surface}")
        model = self.model

        def make(k):
            def f_k(X):
                with torch.no_grad():
                    post = model.models[k].posterior(X)
                    if surface == "mean":
                        values = post.mean.reshape(-1)
                    else:
                        values = post.variance.reshape(-1).clamp(min=1e-12).sqrt()
                return values
            return f_k

        return [make(k) for k in range(self.nobj)]

    def _front(self, surface: str):
        return pymoo_front_on_fns(
            self._posterior_fns(surface), self.d, self.nobj,
            self.npop, self.ngen, dtype=self.dtype, return_population=True,
        )

    def _uniform_from_front(
        self, X_pf: torch.Tensor, X_pop: torch.Tensor
    ) -> torch.Tensor:
        if X_pf.ndim != 2 or X_pf.shape[0] == 0:
            raise RuntimeError("epsilon-Greedy-MOBO received an empty Pareto set")
        if not torch.isfinite(X_pf).all():
            raise RuntimeError("epsilon-Greedy-MOBO Pareto set contains NaN/Inf")
        selected = X_pf[torch.randperm(X_pf.shape[0])[: self.q]]
        return _complete_batch_from_population(
            selected, X_pop, self.observed_x, self.q, self.d
        )

    def suggest(self) -> torch.Tensor:
        if self.policy == "mean":
            surface = "mean"
            log("  [Greedy-MOBO] FULL EXPLOIT: random member of mean PF")
        elif self.policy == "std":
            surface = "std"
            log("  [Greedy-MOBO] FULL EXPLORE: random member of std PF")
        else:
            u = float(torch.rand(1))
            surface = "std" if u <= self.epsilon else "mean"
            branch = "EXPLORE/std-PF" if surface == "std" else "EXPLOIT/mean-PF"
            log(f"  [epsilon-Greedy-MOBO] {branch} (u={u:.3f})")

        X_pf, _, X_pop = self._front(surface)
        return self._uniform_from_front(X_pf, X_pop)


# ============================================================================
# PART 2 — BoTorch-native baselines (merged from mobo_baselines.py)
# ============================================================================

import math
import torch
import numpy as np
from typing import Optional, List

from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
)
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
try:
    from botorch.acquisition.multi_objective.parego import qLogNParEGO
except ImportError:      # older botorch: module path doesn't exist
    qLogNParEGO = None   # ParEGO baseline unavailable; everything else works

from gpytorch.mlls import ExactMarginalLogLikelihood

DEVICE = torch.device("cpu")   # DTYPE already defined above


# ============================================================
# GP MODEL FITTING
# ============================================================
def fit_gp_models(
    train_x: torch.Tensor,   # (N, d) in [0,1]^d
    train_y: torch.Tensor,   # (N, K) or (N,), maximization convention
    nobj: Optional[int] = None,
) -> ModelListGP:
    """Fit one independent ``SingleTaskGP`` per objective.

    The number of objectives is inferred from ``train_y``. The optional
    ``nobj`` argument is retained for compatibility with the existing runner;
    when supplied, it is checked against the number of output columns rather
    than used to slice or discard data.

    The model keeps the normal ``SingleTaskGP`` mean, likelihood, outcome
    transform, priors, and SciPy fitting defaults. Only the covariance family
    is changed from RBF to Matérn-5/2. Each objective is fitted independently
    with up to 10 fitting attempts, retaining the best successful state.
    """
    X = train_x.to(dtype=DTYPE)
    Y = train_y.to(dtype=DTYPE)

    if X.ndim != 2:
        raise ValueError(
            f"train_x must have shape (N, d); got {tuple(X.shape)}"
        )

    # Accept a one-dimensional output from a single-objective test function.
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    elif Y.ndim != 2:
        raise ValueError(
            f"train_y must have shape (N,) or (N, K); got {tuple(Y.shape)}"
        )

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "train_x and train_y contain different numbers of observations: "
            f"X={tuple(X.shape)}, Y={tuple(Y.shape)}"
        )

    if not torch.isfinite(X).all():
        raise RuntimeError("GP training inputs contain NaN or Inf")
    if not torch.isfinite(Y).all():
        raise RuntimeError("GP training outputs contain NaN or Inf")

    inferred_nobj = int(Y.shape[1])
    if nobj is None:
        nobj = inferred_nobj
    elif int(nobj) != inferred_nobj:
        raise ValueError(
            f"nobj={nobj}, but train_y contains {inferred_nobj} "
            "objective column(s)"
        )
    else:
        nobj = int(nobj)

    d = X.shape[-1]
    models = []
    for k in range(nobj):
        # Use the current SingleTaskGP covariance helper and change only the
        # kernel family from its default RBF to Matérn-5/2.
        covar_module = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d,
            use_rbf_kernel=False,
        )
        gp = SingleTaskGP(
            train_X=X,
            train_Y=Y[:, k:k + 1],
            covar_module=covar_module,
        )

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(
            mll,
            max_attempts=10,
            pick_best_of_all_attempts=True,
        )
        gp.eval()
        models.append(gp)

    return ModelListGP(*models)


# ============================================================
# ParEGO  (qLogNParEGO from BoTorch)
# ============================================================
def _parego_candidate(
    model: ModelListGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds_unit: torch.Tensor,
    q: int = 1,
) -> torch.Tensor:
    """Generate candidates using BoTorch qLogNParEGO.

    Research basis: ParEGO random Tchebycheff scalarization
    [knowles2006parego], broader random-scalarization MOBO
    [paria2020flexible], BoTorch [balandat2020botorch], and the LogEI
    numerical reformulation [ament2023logei].
    """
    if qLogNParEGO is None:
        raise ImportError("qLogNParEGO requires a newer botorch; "
                          "upgrade botorch to use the ParEGO baseline.")
    from botorch.optim import optimize_acqf

    acq = qLogNParEGO(
        model=model,
        X_baseline=train_x,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds_unit,
        q=q,
        num_restarts=10,
        raw_samples=512,
    )
    return candidates


# ============================================================
# EHVI  (qNoisyExpectedHypervolumeImprovement from BoTorch)
# ============================================================
def _ehvi_candidate(
    model: ModelListGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds_unit: torch.Tensor,
    ref_point: torch.Tensor,
    q: int = 1,
) -> torch.Tensor:
    """Generate candidate using qNoisyExpectedHypervolumeImprovement.
    References: qEHVI [daulton2020qehvi]; qNEHVI [daulton2021qnehvi];
    hypervolume indicator [zitzler1999hypervolume]."""
    from botorch.optim import optimize_acqf

    acq = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=train_x,
        prune_baseline=True,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds_unit,
        q=q,
        num_restarts=10,
        raw_samples=512,
    )
    return candidates


# ============================================================
# qPOTS  (Reparametrization trick + NSGA-II + maximin)
# ============================================================
def _qpots_candidate(
    model: ModelListGP,
    train_x: torch.Tensor,
    bounds_unit: torch.Tensor,
    nobj: int,
    q: int = 1,
    ngen: int = 20,
    npop: int = 100,
) -> torch.Tensor:
    """
    qPOTS: Draw ONE fixed Thompson sample path per objective using BoTorch
    pathwise posterior sampling, find its Pareto front with NSGA-II, select via
    maximin distance in decision space.
    References: qPOTS [renganathan2025qpots]; pathwise GP sampling
    [wilson2020efficiently]; Thompson sampling [russo2014learning];
    NSGA-II [deb2002nsga2].
    """
    d = bounds_unit.shape[1]
    paths = draw_posterior_paths(model, nobj)
    X_pareto, _, X_pop = pymoo_front_on_fns(
        paths, d, nobj, npop, ngen, return_population=True
    )
    if X_pareto.shape[0] == 0:
        raise RuntimeError("qPOTS baseline received an empty Pareto set")
    indices = _maximin_from_observed(X_pareto, train_x, q)
    initial = X_pareto[indices] if indices.numel() else X_pareto[:0]
    return _complete_batch_from_population(
        initial, X_pop, train_x, q, d
    )


# ============================================================
# TSEMO-HVI  (Spectral sampling + NSGA-II + original HVI selection)
# ============================================================
def _tsemo_candidate(
    model: ModelListGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds_unit: torch.Tensor,
    nobj: int,
    q: int = 1,
    ngen: int = 20,
    npop: int = 100,
    n_features: int = 512,
) -> torch.Tensor:
    """TSEMO-HVI baseline matching Bradford et al. (2018).

    Draw one fixed spectral posterior sample per objective, optimize the
    sample paths with NSGA-II, then choose the candidate with the largest
    sequential hypervolume improvement relative to the observed outcomes.
    Observed and sampled values are both on each GP's standardized scale.
    """
    samplers = []
    for k in range(nobj):
        gp = model.models[k]
        ls, os_, nv, ktype, nu = _extract_gp_hyperparams(gp)
        target_k = gp.train_targets.detach().reshape(-1).to(dtype=DTYPE)
        samplers.append(
            SpectralSampler(
                train_x.to(dtype=DTYPE),
                target_k,
                ls, os_, nv, n_features,
                kernel_type=ktype, nu=nu,
            )
        )

    betas = [sampler.sample_path() for sampler in samplers]
    fns = [
        (lambda sampler_, beta_: (lambda X: sampler_.evaluate(beta_, X)))(
            sampler, beta
        )
        for sampler, beta in zip(samplers, betas)
    ]
    X_pareto, Y_pareto, X_pop = pymoo_front_on_fns(
        fns, bounds_unit.shape[1], nobj, npop, ngen, dtype=DTYPE,
        return_population=True,
    )
    if X_pareto.shape[0] == 0:
        raise RuntimeError("TSEMO-HVI baseline received an empty Pareto set")
    observed_y_std = torch.stack(
        [model.models[k].train_targets.detach().reshape(-1).to(dtype=DTYPE)
         for k in range(nobj)],
        dim=-1,
    )
    return select_from_pareto_front(
        X_pareto, Y_pareto, q, SelectionCriteria.HVI,
        observed_x=train_x, observed_y=observed_y_std, population_x=X_pop,
    )



# ============================================================
# MAIN DISPATCHER
# ============================================================
def pick_candidate_botorch(
    method: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,     # real-scale bounds (2, d)
    nobj: int,
    q: int = 1,
    ref_point: Optional[torch.Tensor] = None,
    ngen: int = 20,
    npop: int = 100,
    tsemo_np: int = 512,
    ucb_beta: float = 2.0,
    # For epsilon-qPOTS variants — pass these through
    epsilon_qpots_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Generate next candidate(s) for any MOBO method.

    All inputs/outputs are in NORMALIZED [0,1]^d space.
    Returns: (q, d) tensor in [0,1]^d
    """
    d = bounds.shape[1]
    bounds_unit = torch.stack([
        torch.zeros(d, dtype=DTYPE),
        torch.ones(d, dtype=DTYPE),
    ])

    # --- Sobol (random) ---
    if method == "Sobol":
        return torch.rand(q, d, dtype=DTYPE)

    # --- Fit GP models ---
    model = fit_gp_models(train_x, train_y[:, :nobj], nobj)

    # --- ParEGO ---
    if method == "ParEGO":
        return _parego_candidate(model, train_x, train_y[:, :nobj],
                                 bounds_unit, q)

    # --- EHVI ---
    if method == "EHVI":
        if ref_point is None:
            raise ValueError("EHVI requires ref_point")
        return _ehvi_candidate(model, train_x, train_y[:, :nobj],
                               bounds_unit, ref_point, q)

    # --- qPOTS ---
    if method == "qPOTS":
        return _qpots_candidate(model, train_x, bounds_unit, nobj, q, ngen, npop)

    # --- TSEMO-HVI (original TSEMO selection rule) ---
    if method == "TSEMO-HVI":
        return _tsemo_candidate(model, train_x, train_y[:, :nobj],
                                bounds_unit, nobj, q, ngen, npop,
                                n_features=tsemo_np)

    # --- UCB (Multi-objective Upper Confidence Bound) ---
    if method == "UCB":
        return _ucb_candidate(model, train_x, bounds_unit, nobj,
                              q, ngen, npop, beta=ucb_beta)

    raise ValueError(f"Unknown method: {method}. "
                     f"Use: ParEGO, EHVI, qPOTS, TSEMO-HVI, UCB, Sobol")


# ============================================================
# UCB  (Multi-objective UCB + NSGA-II + HV-based selection)
# ============================================================
def _ucb_candidate(
    model: ModelListGP,
    train_x: torch.Tensor,
    bounds_unit: torch.Tensor,
    nobj: int,
    q: int = 1,
    ngen: int = 20,
    npop: int = 100,
    beta: float = 2.0,
) -> torch.Tensor:
    """MO-UCB-HVC baseline using the shared pymoo NSGA-II.

    Its first stage matches the USeMO framework idea of solving a cheap MO
    problem over one acquisition surface per objective. Its second stage is
    our HVC case-study selector, not USeMO's uncertainty-hyperrectangle rule.
    """
    def make_ucb_fn(gp):
        def fn(X):
            with torch.no_grad():
                posterior = gp.posterior(X)
                mu = posterior.mean.reshape(-1)
                sigma = posterior.variance.reshape(-1).clamp(min=1e-12).sqrt()
            return mu + beta * sigma
        return fn

    fns = [make_ucb_fn(model.models[k]) for k in range(nobj)]
    X_pareto, Y_pareto, X_pop = pymoo_front_on_fns(
        fns, bounds_unit.shape[1], nobj, npop, ngen, dtype=DTYPE,
        return_population=True,
    )
    if X_pareto.shape[0] == 0:
        raise RuntimeError("UCB baseline received an empty Pareto set")
    return select_from_pareto_front(
        X_pareto, Y_pareto, q, SelectionCriteria.HVC,
        observed_x=train_x, population_x=X_pop,
    )


# ============================================================
# CONVENIENCE: Full BO loop
# ============================================================
def run_bo_loop(*args, **kwargs):
    """Deprecated.

    Use run_demo.py, which enforces the minimization to maximization sign
    boundary and computes reporting HV from observed objective vectors.
    """
    raise RuntimeError(
        "run_bo_loop is deprecated; use run_demo.py"
    )


# ============================================================
# SPECTRAL SANITY CHECK — reference: [do2024epsilon] Figure 2,
# which validates the RFF covariance approximation against the
# analytic kernel in exactly this way.
# ============================================================
def check_spectral_kernel_match(lengthscale=0.7, nu=2.5, n_features=100_000,
                                d=1, dtype=torch.double, verbose=True):
    """Verify that the spectral-weight sampling reconstructs the intended
    kernel: compares the RFF covariance estimate E[cos(w*r)] against the
    analytic Matern-nu (and the SE scheme, for contrast) over a range of
    distances r. Run this once after any change to the spectral sampling
    code, BEFORE launching full benchmarks.

    Returns max absolute error vs the analytic Matern-5/2 kernel
    (should be ~1e-2 or less for n_features=1e5)."""
    import math as _math

    ls = torch.tensor(lengthscale, dtype=dtype)
    df = 2.0 * nu
    z = torch.randn(n_features, d, dtype=dtype)
    u = torch.distributions.Chi2(torch.tensor(df, dtype=dtype)).sample((n_features, 1))
    W_matern = (z / ls) / torch.sqrt(u / df)
    W_se = torch.randn(n_features, d, dtype=dtype) / ls

    def matern52_true(r):
        a = _math.sqrt(5.0) * r / float(ls)
        return (1.0 + a + a * a / 3.0) * _math.exp(-a)

    max_err = 0.0
    rows = []
    for r in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        k_true = matern52_true(r)
        k_rff = float(torch.cos(W_matern[:, 0] * r).mean())
        k_rff_se = float(torch.cos(W_se[:, 0] * r).mean())
        max_err = max(max_err, abs(k_true - k_rff))
        rows.append((r, k_true, k_rff, k_rff_se))

    if verbose:
        print(f"{'r':>5} {'Matern52 true':>14} {'RFF(matern)':>12} {'RFF(se)':>10}")
        for r, kt, km, kse in rows:
            print(f"{r:5.1f} {kt:14.5f} {km:12.5f} {kse:10.5f}")
        print(f"max |true - RFF(matern)| = {max_err:.5f}")
    return max_err


if __name__ == "__main__":
    print("=" * 60)
    print("epsMOBO_criteria - spectral kernel-match smoke test")
    print("=" * 60)
    err = check_spectral_kernel_match()
    assert err < 0.02, f"Spectral sampling does NOT match Matern-5/2 (err={err})"
    print("\nOK: SpectralSampler(matern, nu=2.5) reconstructs Matern-5/2.")
