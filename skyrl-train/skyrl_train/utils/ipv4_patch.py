"""IPv4 hostname patch for HPC clusters.

On some HPC clusters (e.g., Jupiter at JSC), socket.gethostname() returns an
InfiniBand hostname (e.g., "jpbo-113-36-interconnect-1") that resolves to IPv6
addresses. PyTorch's c10d uses this hostname during process group bootstrapping,
and fails with errno 97 (EAFNOSUPPORT) when IPv6 is not supported.

This module provides a patch that forces socket.gethostname() to return an IPv4
address instead, which c10d can use directly without DNS resolution.

Usage:
    from skyrl_train.utils.ipv4_patch import enable_ipv4_hostname_patch

    # Call early in initialization, before any c10d operations
    enable_ipv4_hostname_patch()
"""

import logging
import socket

_original_gethostname = socket.gethostname
_ipv4_hostname_override = None
_patch_enabled = False


def _get_ipv4_hostname():
    """Get an IPv4 address to use as hostname, falling back to original hostname."""
    global _ipv4_hostname_override
    if _ipv4_hostname_override is not None:
        return _ipv4_hostname_override
    return _original_gethostname()


def enable_ipv4_hostname_patch():
    """Enable the IPv4 hostname patch if running in a Ray environment.

    This patches socket.gethostname() to return the IPv4 address from Ray's
    node IP instead of the system hostname. This is safe to call multiple times;
    subsequent calls are no-ops.

    The patch is only applied if:
    1. Ray is initialized and has a node IP address
    2. The node IP is a valid IPv4 address
    """
    global _ipv4_hostname_override, _patch_enabled

    if _patch_enabled:
        return  # Already enabled

    try:
        import ray

        # Check if Ray has a node IP address we can use
        global_node = ray._private.worker._global_node
        if global_node and global_node.node_ip_address:
            ipv4_addr = global_node.node_ip_address
            # Verify it's a valid IPv4 address
            socket.inet_aton(ipv4_addr)  # Raises on invalid IPv4
            _ipv4_hostname_override = ipv4_addr
            socket.gethostname = _get_ipv4_hostname
            _patch_enabled = True
            logging.info(f"[ipv4-patch] Patched socket.gethostname() to return {ipv4_addr}")
    except ImportError:
        logging.debug("[ipv4-patch] Ray not available, skipping IPv4 hostname patch")
    except Exception as e:
        logging.debug(f"[ipv4-patch] Could not enable IPv4 hostname patch: {e}")


def get_ipv4_address():
    """Get the IPv4 address that would be used by the patch, or None if not available."""
    global _ipv4_hostname_override
    return _ipv4_hostname_override


def is_patch_enabled():
    """Check if the IPv4 hostname patch is currently enabled."""
    global _patch_enabled
    return _patch_enabled
