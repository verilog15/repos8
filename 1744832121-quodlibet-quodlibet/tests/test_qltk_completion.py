# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from tests import TestCase

from gi.repository import Gtk

from quodlibet import config
from quodlibet.library import SongLibrary
from quodlibet.qltk.completion import EntryWordCompletion, LibraryTagCompletion
from quodlibet.qltk.completion import LibraryValueCompletion


class TEntryWordCompletion(TestCase):
    def test_ctr(self):
        w = EntryWordCompletion()
        e = Gtk.Entry()
        e.set_completion(w)
        self.assertEqual(w.get_entry(), e)
        self.assertEqual(e.get_completion(), w)
        e.destroy()


class TLibraryTagCompletion(TestCase):
    def test_ctr(self):
        w = LibraryTagCompletion(SongLibrary())
        e = Gtk.Entry()
        e.set_completion(w)
        self.assertEqual(w.get_entry(), e)
        self.assertEqual(e.get_completion(), w)
        e.destroy()


class TLibraryValueCompletion(TestCase):
    def setUp(self):
        config.init()

    def tearDown(self):
        config.quit()

    def test_ctr(self):
        w = LibraryValueCompletion("artist", SongLibrary())
        e = Gtk.Entry()
        e.set_completion(w)
        self.assertEqual(w.get_entry(), e)
        self.assertEqual(e.get_completion(), w)
        e.destroy()
