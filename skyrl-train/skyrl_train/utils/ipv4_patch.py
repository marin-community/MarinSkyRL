"""IPv4 hostname and DNS patch for HPC clusters.

On some HPC clusters (e.g., Jupiter at JSC), socket.gethostname() returns an
InfiniBand hostname (e.g., "jpbo-113-36-interconnect-1") that resolves to IPv6
addresses. PyTorch's c10d uses this hostname during process group bootstrapping,
and fails with errno 97 (EAFNOSUPPORT) when IPv6 is not supported.

This module provides TWO patches:
1. socket.gethostname() - returns IPv4 address instead of hostname
2. socket.getaddrinfo() - forces AF_INET (IPv4) for hostname resolution

Both patches are needed because c10d processes exchange hostnames with each other,
and even if node A returns an IPv4 address for its hostname, node B may still
send its IB hostname which then needs to be resolved.

Usage:
    from skyrl_train.utils.ipv4_patch import enable_ipv4_hostname_patch

    # Call early in initialization, before any c10d operations
    enable_ipv4_hostname_patch()
"""

import logging
import socket

_original_gethostname = socket.gethostname
_original_getaddrinfo = socket.getaddrinfo
_ipv4_hostname_override = None
_patch_enabled = False


def _get_ipv4_hostname():
    """Get an IPv4 address to use as hostname, falling back to original hostname."""
    global _ipv4_hostname_override
    if _ipv4_hostname_override is not None:
        return _ipv4_hostname_override
    return _original_gethostname()


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Wrapper around getaddrinfo that forces IPv4 (AF_INET).

    If no address family is specified (family=0), we force AF_INET to prevent
    IPv6 addresses from being returned. This fixes c10d connection failures
    on clusters where hostnames resolve to both IPv4 and IPv6 but IPv6 is
    not supported by the network stack.
    """
    # If caller didn't specify a family, force IPv4
    if family == 0:
        family = socket.AF_INET

    try:
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        # If IPv4 resolution fails, try without forcing IPv4
        # (some addresses may be IPv6-only)
        if family == socket.AF_INET:
            logging.debug(f"[ipv4-patch] IPv4 resolution failed for {host}, trying any family")
            return _original_getaddrinfo(host, port, 0, type, proto, flags)
        raise


def enable_ipv4_hostname_patch():
    """Enable the IPv4 hostname and DNS resolution patch.

    This patches:
    1. socket.gethostname() - returns IPv4 address from Ray instead of hostname
    2. socket.getaddrinfo() - forces AF_INET (IPv4) for hostname resolution

    This is safe to call multiple times; subsequent calls are no-ops.

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

            # Patch gethostname
            _ipv4_hostname_override = ipv4_addr
            socket.gethostname = _get_ipv4_hostname

            # Patch getaddrinfo to force IPv4
            socket.getaddrinfo = _ipv4_getaddrinfo

            _patch_enabled = True
            logging.info(f"[ipv4-patch] Patched socket.gethostname() to return {ipv4_addr}")
            logging.info("[ipv4-patch] Patched socket.getaddrinfo() to force IPv4")
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
