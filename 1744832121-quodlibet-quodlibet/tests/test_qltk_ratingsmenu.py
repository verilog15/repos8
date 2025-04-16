# Copyright 2012,2013 Christoph Reiter
#                2016 Nick Boultbee
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gi.repository import Gtk
from senf import fsnative

from tests import TestCase
from quodlibet import config
from quodlibet.qltk.ratingsmenu import RatingsMenuItem
from quodlibet.library import SongLibrary, SongLibrarian
from quodlibet.formats import AudioFile

NUM_RATINGS = 4


class TRatingsMenuItem(TestCase):
    def setUp(self):
        config.RATINGS = config.HardCodedRatingsPrefs()
        self.assertEqual(config.RATINGS.number, NUM_RATINGS)
        self.library = SongLibrary()
        self.library.librarian = SongLibrarian()
        self.af = AudioFile({"~filename": fsnative("/foo"), "~#rating": 1.0})
        self.af.sanitize()
        self.rmi = RatingsMenuItem([self.af], self.library)

    def tearDown(self):
        self.rmi.destroy()
        self.library.destroy()
        self.library.librarian.destroy()

    def test_menuitem_children(self):
        children = [
            mi
            for mi in self.rmi.get_submenu().get_children()
            if isinstance(mi, Gtk.CheckMenuItem)
        ]
        self.assertEqual(len(children), NUM_RATINGS + 1)
        highest = children[-1]
        self.assertEqual(highest.get_active(), True)
        self.assertEqual(children[1].get_active(), False)

    def test_no_rating(self):
        af = AudioFile({"~filename": fsnative("/foobar"), "artist": "foo"})
        rmi = RatingsMenuItem([af], self.library)
        children = [
            mi
            for mi in rmi.get_submenu().get_children()
            if isinstance(mi, Gtk.CheckMenuItem)
        ]
        assert not any(c.get_active() for c in children)

    def test_set_remove_rating(self):
        self.rmi.set_rating(0.5, [self.af], self.library)
        assert self.af.has_rating
        self.assertEqual(self.af("~#rating"), 0.5)
        self.rmi.remove_rating([self.af], self.library)
        assert not self.af.has_rating
