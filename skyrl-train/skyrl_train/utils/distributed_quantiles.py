"""Exact float32 absolute quantiles with bounded retention and radix selection."""

from collections.abc import Callable
import math

import torch


class AbsoluteQuantileBuffer:
    """Retain at most ``capacity`` finite float32 values per token-owning rank.

    Four radix rounds communicate four 256-bin int64 histograms (8192 bytes
    per round), independent of token count. Each rank scans local chunks;
    no token vector is sent to another rank. Overflow invalidates the result.
    """

    def __init__(self, device: torch.device, capacity: int = 4_194_304):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Quantile capacity must be a positive integer")
        self.device = device
        self.capacity = capacity
        self.chunks: list[torch.Tensor] = []
        self.selected = 0
        self.finite = 0
        self.overflow = False
        self.nonrepresentable = 0

    def add(self, selected_delta: torch.Tensor) -> None:
        values = selected_delta.detach().reshape(-1)
        self.selected += values.numel()
        values = values[torch.isfinite(values)].abs().float()
        self.finite += values.numel()
        self.nonrepresentable += int((~torch.isfinite(values)).sum().item())
        if self.overflow or self.finite > self.capacity:
            self.overflow = True
            self.chunks.clear()
        elif values.numel():
            # Own compact storage, so a small view cannot retain a full batch.
            self.chunks.extend(chunk.clone() for chunk in values.split(65_536))

    @property
    def retained_bytes(self) -> int:
        return sum(chunk.numel() * chunk.element_size() for chunk in self.chunks)

    def quantiles(
        self,
        sum_reduce: Callable[[torch.Tensor], torch.Tensor] = lambda value: value,
        *,
        owns_tokens: bool = True,
    ) -> dict[str, float]:
        """Pool counts then select p50/p95, matching linear torch.quantile.

        ``sum_reduce`` must SUM over distinct-token ownership ranks and return
        the same tensor on every participating rank. Replicas still participate
        with zero counts. Empty and overflow ranks take identical collectives.
        """
        header = torch.tensor(
            [self.selected, self.finite, int(self.overflow), self.nonrepresentable] if owns_tokens else [0, 0, 0, 0],
            device=self.device,
            dtype=torch.int64,
        )
        selected, finite, overflow, nonrepresentable = sum_reduce(header).tolist()
        result = {
            "selected_tokens": float(selected),
            "finite_tokens": float(finite),
            "finite_fraction": finite / selected if selected else 0.0,
            "quantiles_valid": float(finite > 0 and not overflow and not nonrepresentable),
            "quantiles_overflow": float(overflow > 0),
            "quantiles_overflow_ranks": float(overflow),
            "quantiles_nonrepresentable_tokens": float(nonrepresentable),
            "abs_log_ratio_p50": 0.0,
            "abs_log_ratio_p95": 0.0,
        }
        if not result["quantiles_valid"]:
            return result
        ranks = [(finite - 1) * probability for probability in (0.5, 0.95)]
        remaining = torch.tensor(
            [rank for value in ranks for rank in (math.floor(value), math.ceil(value))],
            device=self.device,
            dtype=torch.int64,
        )
        prefix = torch.zeros(4, device=self.device, dtype=torch.int64)
        chunks = self.chunks if owns_tokens else []
        for shift in (24, 16, 8, 0):
            histogram = torch.zeros((4, 256), device=self.device, dtype=torch.int64)
            for chunk in chunks:
                # Finite absolute float32 values have monotonic positive IEEE
                # bit patterns. Widen only this local chunk for integer shifts.
                bits = chunk.view(torch.int32).to(torch.int64)
                byte = (bits >> shift) & 255
                for target in range(4):
                    matches = (bits >> (shift + 8)) == (prefix[target] >> (shift + 8))
                    histogram[target] += torch.bincount(byte[matches], minlength=256)
            histogram = sum_reduce(histogram)
            cumulative = histogram.cumsum(-1)
            digit = (cumulative <= remaining.unsqueeze(-1)).sum(-1)
            before = cumulative.gather(1, (digit - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1)
            remaining -= torch.where(digit > 0, before, 0)
            prefix |= digit << shift
        endpoints = prefix.to(torch.int32).view(torch.float32).double().tolist()
        for index, name in enumerate(("abs_log_ratio_p50", "abs_log_ratio_p95")):
            fraction = ranks[index] - math.floor(ranks[index])
            result[name] = endpoints[index * 2] * (1 - fraction) + endpoints[index * 2 + 1] * fraction
        return result
