"""Fixed relative INITIAL-weight noise for controlled-mismatch experiments."""

import hashlib

import torch


class FixedSamplerNoise:
    """Retain the initial delta and a clean copy of each current sampler snapshot."""

    def __init__(self, parameters, scale: float, seed: int):
        self.parameters = dict(parameters)
        self.delta = {}
        self.clean = {}
        self._evaluation_snapshot = None
        for name, parameter in self.parameters.items():
            initial = parameter.detach().float().cpu()
            digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
            generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
            self.delta[name] = (scale * torch.randn(initial.shape, generator=generator) * initial).to(parameter.dtype)
        self._evidence = {
            "parameter_count": len(self.delta),
            "delta_squared_norm": sum(float(delta.float().square().sum()) for delta in self.delta.values()),
            "delta_sample_sha256": hashlib.sha256(
                b"".join(delta.flatten()[:16].float().numpy().tobytes() for delta in self.delta.values())
            ).hexdigest(),
        }
        self.after_sync()

    @torch.no_grad()
    def after_sync(self) -> None:
        """Capture fresh unperturbed weights, then install the same fixed delta."""
        self.clean = {name: parameter.detach().cpu().clone() for name, parameter in self.parameters.items()}
        self.set_enabled(True)

    @torch.no_grad()
    def set_enabled(self, enabled: bool) -> None:
        """Restore exact stored clean weights or clean-plus-delta; never subtract rounded noise."""
        for name, parameter in self.parameters.items():
            clean = self.clean[name].to(parameter.device)
            if enabled:
                clean = clean + self.delta[name].to(parameter.device)
            parameter.copy_(clean)

    def evidence(self) -> dict:
        return dict(self._evidence)

    @torch.no_grad()
    def snapshot_for_evaluation(self) -> None:
        """Keep the actual serving weights and their clean base across a fresh-policy evaluation."""
        if self._evaluation_snapshot is not None:
            raise RuntimeError("A sampler evaluation snapshot is already active")
        weights = {name: parameter.detach().cpu().clone() for name, parameter in self.parameters.items()}
        self._evaluation_snapshot = (weights, self.clean)

    @torch.no_grad()
    def restore_after_evaluation(self) -> None:
        """Restore the exact prior serving policy, including its original clean base."""
        if self._evaluation_snapshot is None:
            raise RuntimeError("No sampler evaluation snapshot is active")
        weights, self.clean = self._evaluation_snapshot
        for name, parameter in self.parameters.items():
            parameter.copy_(weights[name].to(parameter.device))
        self._evaluation_snapshot = None
