# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from tests import TestCase, init_fake_app, destroy_fake_app

from quodlibet import app
from quodlibet.mmkeys import MMKeysHandler, iter_backends


class TMmKeys(TestCase):
    def setUp(self):
        init_fake_app()

    def tearDown(self):
        destroy_fake_app()

    def test_handler(self):
        handler = MMKeysHandler(app)
        handler.quit()

    def test_backends(self):
        for backend in iter_backends():
            backend.is_active()
            instance = backend("Foo", lambda action: None)
            instance.grab()
            instance.cancel()
            instance.cancel()
