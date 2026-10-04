"""
The identity of this Jelibox installation: a random UUID ("device_id") created the first time
it is needed, not at install time - re-running the installer must not change it, and a copied
install folder gets its own as soon as it is used on the new machine.

Stored in configs/_app.json. It is a name, not a secret; the secret access key issued by a
server lives in configs/_server.json (see servers.py).
"""
import uuid

from .. import app_settings

_KEY = "device_id"


def is_valid_device_id(value):
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def get_device_id():
    value = app_settings.get(_KEY)
    if is_valid_device_id(value):
        return value
    app_settings.set(_KEY, str(uuid.uuid4()))
    # Another Jelibox process may have created one in the meantime - the stored value wins.
    return app_settings.get(_KEY)


def reset_device_id():
    """Give this installation a new identity (it will have to be approved again by every server)."""
    app_settings.set(_KEY, str(uuid.uuid4()))
    return app_settings.get(_KEY)
