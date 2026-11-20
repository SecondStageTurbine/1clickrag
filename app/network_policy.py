# SPDX-License-Identifier: MPL-2.0
"""Fail-closed network policy for disconnected and classified deployments."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


class NetworkPolicyError(ValueError):
    """A configured endpoint is outside the permitted enclave boundary."""


_BUILTIN_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "qdrant",
    "ollama",
    "host.docker.internal",
}
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(block)
    for block in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)


def is_internal_host(host: str, allowed_hosts: list[str] | tuple[str, ...] = ()) -> bool:
    """Return whether *host* is unambiguously local or explicitly allowed.

    DNS names are not resolved here: resolution could itself leave the enclave,
    and an address can change between validation and use. Internal FQDNs must be
    named in ``RAG_DARK_NETWORK_ALLOWED_HOSTS``.
    """
    candidate = (host or "").strip().lower().rstrip(".")
    allowed = {item.strip().lower().rstrip(".") for item in allowed_hosts if item.strip()}
    if candidate in _BUILTIN_HOSTS or candidate in allowed:
        return True
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return candidate.endswith((".localhost", ".local", ".lan"))
    return (
        address.is_loopback
        or address.is_link_local
        or any(address in network for network in _PRIVATE_NETWORKS)
    )


def require_internal_url(url: str, purpose: str,
                         allowed_hosts: list[str] | tuple[str, ...] = ()) -> None:
    """Reject malformed or non-enclave HTTP(S) endpoints."""
    try:
        parsed = urlsplit(url)
        port = parsed.port  # force validation of a malformed port
    except ValueError as exc:
        raise NetworkPolicyError(f"invalid {purpose} URL: {url!r}") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise NetworkPolicyError(
            f"{purpose} must be an explicit http(s) URL in dark-network mode"
        )
    del port
    if not is_internal_host(parsed.hostname, allowed_hosts):
        raise NetworkPolicyError(
            f"dark-network mode blocked {purpose} host {parsed.hostname!r}; "
            "use an enclave-local address or add the exact internal FQDN to "
            "RAG_DARK_NETWORK_ALLOWED_HOSTS"
        )


def require_loopback_bind(host: str, allow_remote: bool = False) -> None:
    """Keep the unauthenticated API on this host unless explicitly fronted."""
    if allow_remote:
        return
    candidate = (host or "").strip().lower()
    if candidate not in ("localhost", "127.0.0.1", "::1"):
        raise NetworkPolicyError(
            "dark-network mode requires RAG_HOST=127.0.0.1 because the API has "
            "no built-in user authentication; set RAG_DARK_NETWORK_ALLOW_REMOTE=1 "
            "only behind an approved authenticated enclave gateway"
        )
