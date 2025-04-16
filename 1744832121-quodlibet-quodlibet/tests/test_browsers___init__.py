# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import sys
from tests import TestCase

from quodlibet import browsers

browsers.init()


class TBrowsers(TestCase):
    def test_presence(self):
        assert browsers.tracks
        assert browsers.paned
        assert browsers.iradio
        assert browsers.podcasts
        assert browsers.albums
        assert browsers.playlists
        assert browsers.filesystem

    def test_get(self):
        assert browsers.get("SearchBar") is browsers.tracks.TrackList
        self.assertTrue(browsers.get("FileSystem") is browsers.filesystem.FileSystem)
        self.assertEqual(browsers.get("Paned"), browsers.paned.PanedBrowser)
        self.assertEqual(browsers.get("paned"), browsers.paned.PanedBrowser)
        self.assertEqual(browsers.get("panedbrowser"), browsers.paned.PanedBrowser)

    def test_get_podcasts_aka_feeds(self):
        cls = browsers.podcasts.Podcasts
        assert browsers.get("Podcasts") == cls
        assert browsers.get("audiofeeds") == cls, "Legacy name unsupported"

    def test_default(self):
        self.assertEqual(browsers.default, browsers.tracks.TrackList)

    def test_name(self):
        self.assertEqual(browsers.name(browsers.tracks.TrackList), "SearchBar")

    def test_get_invalid(self):
        self.assertRaises(ValueError, browsers.get, "DoesNotExist")

    def test_index(self):
        self.assertEqual(
            browsers.browsers[browsers.index("SearchBar")], browsers.tracks.TrackList
        )
        self.assertEqual(
            browsers.browsers[browsers.index("FileSystem")],
            browsers.filesystem.FileSystem,
        )

    def test_index_invalid(self):
        self.assertRaises(ValueError, browsers.index, "DoesNotExist")

    def test_migrate(self):
        self.assertTrue(sys.modules["browsers.audiofeeds"] is browsers.podcasts)
        self.assertTrue(sys.modules["browsers.iradio"] is browsers.iradio)

    def test_old_names(self):
        self.assertEqual(browsers.get("PanedBrowser"), browsers.get("Paned"))
        self.assertEqual(browsers.get("PlaylistsBrowser"), browsers.get("Playlists"))
