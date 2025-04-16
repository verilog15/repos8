# Copyright 2018 Christoph Reiter
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from quodlibet import print_d
from quodlibet.util import is_linux
from ._base import SessionClient, SessionError


def iter_backends():
    if is_linux():
        from .gnome import GnomeSessionClient

        yield GnomeSessionClient
        from .xfce import XfceSessionClient

        yield XfceSessionClient
        from .xsmp import XSMPSessionClient

        yield XSMPSessionClient
    # dummy one last
    yield SessionClient


def init(app):
    """Returns an active SessionClient instance or None"""

    for backend in iter_backends():
        print_d(f"Trying {backend.__name__}")
        client = backend()
        try:
            client.open(app)
        except SessionError as e:
            print_d(str(e))
        else:
            return client
    raise AssertionError()
