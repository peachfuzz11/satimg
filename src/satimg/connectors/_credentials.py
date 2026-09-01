"""Credential resolution for the download connectors.

Searching a STAC catalogue needs no account; only :meth:`Connector.download`
does. Provide credentials either explicitly::

    Sentinel2Connector(username="me@example.com", password="...")

or leave them unset and export the service's environment variables
(``<PREFIX>_USERNAME`` / ``<PREFIX>_PASSWORD`` / optional ``<PREFIX>_TOKEN``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class CredentialsError(RuntimeError):
    """Raised when a download is attempted without usable credentials."""


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str
    token: str | None = None

    @classmethod
    def resolve(
        cls,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        *,
        service: str,
        env_prefix: str,
    ) -> "Credentials":
        """Explicit args win; anything missing falls back to ``<env_prefix>_*``.

        Raises :class:`CredentialsError` if username or password is still absent.
        """
        username = username or os.environ.get(f"{env_prefix}_USERNAME")
        password = password or os.environ.get(f"{env_prefix}_PASSWORD")
        token = token or os.environ.get(f"{env_prefix}_TOKEN")
        if not (username and password):
            raise CredentialsError(
                f"{service} downloads require a username and password. Pass "
                f"username=/password= to the connector, or set the "
                f"{env_prefix}_USERNAME and {env_prefix}_PASSWORD environment variables."
            )
        return cls(username, password, token)
