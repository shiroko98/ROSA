"""Use credentials for the explicitly selected W&B host, including its port."""
from __future__ import annotations

import netrc
import os
from urllib.parse import urlsplit


def configure_wandb_auth():
    """Work around older SDKs dropping a self-hosted server's netrc port.

    Never print, persist, or forward credentials to any other host. An explicit
    WANDB_API_KEY always wins. With no explicit server, leave SDK defaults alone.
    """
    if os.environ.get('WANDB_API_KEY') or not os.environ.get('WANDB_BASE_URL'):
        return False
    host = urlsplit(os.environ['WANDB_BASE_URL'])
    if host.scheme not in ('http', 'https') or not host.netloc or host.username or host.password:
        raise ValueError('WANDB_BASE_URL must be an HTTP(S) server without embedded credentials.')
    try:
        auth = netrc.netrc().authenticators(host.netloc)
    except (OSError, netrc.NetrcParseError):
        return False
    if not auth or not auth[2]:
        return False
    os.environ['WANDB_API_KEY'] = auth[2]
    return True
