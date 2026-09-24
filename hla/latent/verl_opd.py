"""S6 replay adapter to upstream verl's k1 + vanilla PG-OPD loss.

Uses upstream loss functions, not a reimplementation and not RayPPOTrainer.
S6 owns its C1 forward/TBPTT and SUM gradient reduction. Therefore dp_size=1
below is intentional: loss is divided by the GLOBAL response-token count.
"""
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch

VERL_REVISION = '8050ff113b5b2346e2709ee82c209e03b7c82901'


def installed_revision():
    """Read VCS installation provenance, or the checkout behind editable installs."""
    import verl
    distribution = importlib.metadata.distribution('verl')
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    revision = direct.get('vcs_info', {}).get('commit_id')
    if revision is None:
        root = Path(verl.__file__).resolve().parent.parent
        if (root/'.git').exists():
            revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != VERL_REVISION:
        raise RuntimeError(f'Expected pinned verl {VERL_REVISION}, found {revision!r}; '
                           'install requirements-opd-verl.txt with --no-deps')
    return revision


@dataclass
class VerlOPDLoss:
    clip_ratio: float = .2
    loss_max_clamp: float = 10.

    def __post_init__(self):
        if not 0 < self.clip_ratio < 1 or self.loss_max_clamp <= 0:
            raise ValueError('Invalid OPD clipping settings')
        try:
            from verl.trainer.ppo.core_algos import kl_penalty, compute_policy_loss_vanilla
            from omegaconf import OmegaConf
        except ImportError as error:
            raise ImportError('Install the pinned verl runtime in requirements-opd.txt; '
                              'OPD has no fallback loss implementation') from error
        self.kl_penalty = kl_penalty
        self.policy_loss = compute_policy_loss_vanilla
        self.config = OmegaConf.create(dict(clip_ratio=self.clip_ratio,
            clip_ratio_low=self.clip_ratio, clip_ratio_high=self.clip_ratio,
            clip_ratio_c=3., global_batch_info={}))
        self.version = importlib.metadata.version('verl')
        self.revision = installed_revision()

    def __call__(self, logp, old_logp, teacher_logp, mask, normalizer):
        if normalizer <= 0 or logp.shape != mask.shape or logp.shape != old_logp.shape or logp.shape != teacher_logp.shape:
            raise ValueError('OPD requires aligned response tensors and a global token denominator')
        # Same k1 and detached advantage as verl.trainer.distillation.losses.
        # Upstream log_prob_min_clamp is used by top-k kernels, not its k1 path;
        # do NOT silently clamp sampled-token log-probs or the PPO ratio here.
        signal = self.kl_penalty(logp, teacher_logp.detach(), 'k1')
        advantages = -signal.clamp(-self.loss_max_clamp, self.loss_max_clamp).detach()
        self.config.global_batch_info = dict(dp_size=1, batch_num_tokens=normalizer)
        loss, _ = self.policy_loss(old_log_prob=old_logp.detach(), log_prob=logp,
            advantages=advantages, response_mask=mask, loss_agg_mode='token-mean', config=self.config)
        return loss
