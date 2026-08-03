"""Unit tests for the ``--ingress-mode auto`` derivation in cloud/iris/iris_backend.py.

Proves that controller-ingress is auto-enabled ONLY for the opencode harness on a CoreWeave
target, that an EXPLICIT ``--ingress-mode`` always wins over the ``auto`` derivation, and that
every other harness (terminus-2 etc.) stays on the DIRECT marinskyrl HTTP endpoint. No live
controller / cluster is exercised — this is pure argument-namespace derivation over a small
config-text fixture.

Run:
    python -m pytest cloud/iris/tests/test_ingress_mode_auto.py -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.iris_backend import (  # noqa: E402
    _rl_config_needs_controller_ingress,
    autoconfigure_ingress,
    create_parser,
)

_OPENCODE_CFG = """\
generator:
  harbor:
    harness:
      name: opencode
      max_turns: 40
"""

_TERMINUS2_CFG = """\
generator:
  harbor:
    harness:
      name: terminus_2
      max_turns: 40
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _args(rl_config, ingress_mode="auto", target_cluster="cw-us-east-02a"):
    return argparse.Namespace(
        rl_config=rl_config,
        ingress_mode=ingress_mode,
        target_cluster=target_cluster,
        cluster="",
        ingress_host=None,
    )


# --------------------------------------------------------------------------- #
# _rl_config_needs_controller_ingress: opencode-only detection
# --------------------------------------------------------------------------- #


def test_needs_controller_ingress_true_only_for_opencode(tmp_path):
    assert _rl_config_needs_controller_ingress(_write(tmp_path, "oc.yaml", _OPENCODE_CFG)) is True
    assert _rl_config_needs_controller_ingress(_write(tmp_path, "t2.yaml", _TERMINUS2_CFG)) is False
    assert _rl_config_needs_controller_ingress(None) is False
    assert _rl_config_needs_controller_ingress("/no/such/file.yaml") is False


# --------------------------------------------------------------------------- #
# autoconfigure_ingress: auto derivation + explicit override
# --------------------------------------------------------------------------- #


def test_parser_default_ingress_mode_is_auto():
    args = create_parser().parse_args(["--rl_config", "x", "--model_path", "y"])
    assert args.ingress_mode == "auto"


def test_opencode_auto_derives_controller(tmp_path):
    args = _args(_write(tmp_path, "oc.yaml", _OPENCODE_CFG), ingress_mode="auto")
    autoconfigure_ingress(args)
    assert args.ingress_mode == "controller"


def test_opencode_explicit_direct_wins(tmp_path):
    args = _args(_write(tmp_path, "oc.yaml", _OPENCODE_CFG), ingress_mode="direct")
    autoconfigure_ingress(args)
    assert args.ingress_mode == "direct"


def test_terminus2_auto_derives_direct(tmp_path):
    args = _args(_write(tmp_path, "t2.yaml", _TERMINUS2_CFG), ingress_mode="auto")
    autoconfigure_ingress(args)
    assert args.ingress_mode == "direct"


def test_terminus2_explicit_controller_wins(tmp_path):
    args = _args(_write(tmp_path, "t2.yaml", _TERMINUS2_CFG), ingress_mode="controller")
    autoconfigure_ingress(args)
    assert args.ingress_mode == "controller"
