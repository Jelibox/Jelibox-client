"""
Groundwork for team collaboration through a Jelibox server on the local network.

For now this package only holds what lives on THIS machine: the device identity and the list
of servers the user typed in, plus the credentials stored for them. It contains no network
code on purpose - Jelibox stays fully local and silent until a server is configured and a
later version adds the connection features.

The whole feature is hidden unless enabled(): set the environment variable JELIBOX_COLLAB=1
or "collab_enabled": true in configs/_app.json.
"""
import os

from .. import app_settings


def enabled():
    if os.environ.get("JELIBOX_COLLAB", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return app_settings.get("collab_enabled") is True
