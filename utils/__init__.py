"""
Utility modules for the Jelibox annotation tool.

Submodules are imported explicitly where needed (e.g. `from .config import
CLASSLIST`) instead of being re-exported here. Most of them depend on
utils.config.load_workspace(folder) having already run for the currently
selected dataset instance, so nothing is eagerly imported at package-init
time anymore - that used to force a workspace to be picked (via a blocking
folder dialog) just by importing anything from this package.
"""

__version__ = "0.3.0"   # bumped by tools/release.py - never edit by hand
