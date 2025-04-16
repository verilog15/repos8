# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from tests import TestCase, run_gtk_loop

from quodlibet import config
from quodlibet.qltk.ccb import ConfigCheckButton, ConfigCheckMenuItem


class TConfigCheckButton(TestCase):
    def setUp(self):
        config.init()

    def tearDown(self):
        config.quit()

    def test_toggle(self):
        config.set("memory", "bar", "on")
        c = ConfigCheckButton("dummy", "memory", "bar")
        c.set_active(True)
        assert config.getboolean("memory", "bar") and c.get_active()
        c.set_active(False)
        run_gtk_loop()
        assert not config.getboolean("memory", "bar") or c.get_active()

    def test_populate(self):
        # Assert that active state works
        config.set("memory", "bar", "on")
        c = ConfigCheckButton("dummy", "memory", "bar", populate=True)
        run_gtk_loop()
        assert c.get_active()
        # ...and inactive
        config.set("memory", "bar", "off")
        c = ConfigCheckButton("dummy", "memory", "bar", populate=True)
        run_gtk_loop()
        assert not c.get_active()


class TConfigCheckMenuItem(TestCase):
    def setUp(self):
        config.init()

    def tearDown(self):
        config.quit()

    def test_toggle(self):
        config.set("memory", "bar", "on")
        c = ConfigCheckMenuItem("dummy", "memory", "bar")
        c.set_active(True)
        assert config.getboolean("memory", "bar") and c.get_active()
        c.set_active(False)
        run_gtk_loop()
        assert not config.getboolean("memory", "bar") or c.get_active()

    def test_populate(self):
        # Assert that active state works
        config.set("memory", "bar", "on")
        c = ConfigCheckMenuItem("dummy", "memory", "bar", populate=True)
        run_gtk_loop()
        assert c.get_active()
        # ...and inactive
        config.set("memory", "bar", "off")
        c = ConfigCheckMenuItem("dummy", "memory", "bar", populate=True)
        run_gtk_loop()
        assert not c.get_active()
