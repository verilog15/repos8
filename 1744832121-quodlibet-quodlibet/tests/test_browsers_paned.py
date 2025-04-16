# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from tests import TestCase, run_gtk_loop
from .helper import realized

from gi.repository import Gtk
from senf import fsnative

from quodlibet import config, util

from quodlibet.browsers.paned import PanedBrowser
from quodlibet.browsers.paned.util import PaneConfig
from quodlibet.browsers.paned.util import get_headers
from quodlibet.browsers.paned.models import AllEntry, UnknownEntry, SongsEntry
from quodlibet.browsers.paned.models import PaneModel
from quodlibet.browsers.paned.prefs import PatternEditor, Preferences
from quodlibet.browsers.paned.prefs import PreferencesButton, ColumnMode
from quodlibet.browsers.paned.pane import Pane
from quodlibet.formats import AudioFile
from quodlibet.util.collection import Collection
from quodlibet.util.string.date import format_date
from quodlibet.library import SongLibrary, SongLibrarian

SONGS = [
    AudioFile(
        {
            "title": "three",
            "artist": "<boris>",
            "genre": "Rock",
            "~filename": fsnative("/bin/ls"),
            "foo": "bar",
        }
    ),
    AudioFile(
        {
            "title": "two",
            "artist": "mu",
            "genre": "Rock",
            "~filename": fsnative("/dev/zero"),
            "foo": "bar",
        }
    ),
    AudioFile(
        {
            "title": "four",
            "artist": "piman",
            "genre": "J-Pop",
            "~filename": fsnative("/dev/null"),
            "foo": "bar\nquux",
        }
    ),
    AudioFile(
        {
            "title": "one",
            "artist": "piman",
            "genre": "J-Pop",
            "~filename": fsnative("/bin/foo"),
            "foo": "bar\nnope",
        }
    ),
    AudioFile({"title": "xxx", "~filename": fsnative("/bin/bar"), "foo": "bar"}),
]

UNKNOWN_ARTIST = AudioFile(dict(SONGS[0]))
del UNKNOWN_ARTIST["artist"]

ALBUM = Collection()
ALBUM.songs = SONGS


class TPanedBrowser(TestCase):
    Bar = PanedBrowser

    def setUp(self):
        config.init()
        config.set("browsers", "panes", "artist")
        library = SongLibrary()
        library.librarian = SongLibrarian()
        PanedBrowser.init(library)
        for af in SONGS:
            af.sanitize()
        library.add(SONGS)
        self.bar = self.Bar(library)

        self.last = None
        self.emit_count = 0

        def selected_cb(browser, songs, *args):
            self.last = list(songs)
            self.emit_count += 1

        self.bar.connect("songs-selected", selected_cb)

    def test_get_set_headers(self):
        config.set("browsers", "panes", "~people album")
        self.assertEqual(get_headers(), ["~people", "album"])

    def test_pack(self):
        to_pack = Gtk.Button()
        container = self.bar.pack(to_pack)
        self.bar.unpack(container, to_pack)

    def test_can_filter(self):
        for key in ["foo", "title", "fake~key", "~woobar", "~#huh"]:
            assert not self.bar.can_filter_tag(key)
        assert self.bar.can_filter("artist")
        assert self.bar.can_filter_text()

    def test_filter_text(self):
        self.bar.activate()

        self.bar.filter_text("artist=nope")
        run_gtk_loop()
        self.assertEqual(set(self.last), set())

        self.bar.filter_text("artist=!<boris>")
        run_gtk_loop()
        self.assertEqual(set(self.last), set(SONGS[1:]))

    def test_filter_value(self):
        self.bar.activate()
        expected = [SONGS[0]]
        self.bar.filter("artist", ["<boris>"])
        run_gtk_loop()
        self.assertEqual(self.last, expected)

    def test_filter_notvalue(self):
        self.bar.activate()
        expected = SONGS[1:4]
        self.bar.filter("artist", ["notvalue", "mu", "piman"])
        run_gtk_loop()
        self.assertEqual(set(self.last), set(expected))

    def test_restore(self):
        config.set("browsers", "query_text", "foo")
        self.bar.restore()
        self.assertEqual(self.bar._get_text(), "foo")
        self.bar.finalize(True)
        run_gtk_loop()
        self.assertEqual(self.emit_count, 0)

    def test_numeric_config_search(self):
        config.set("browsers", "panes", "~#track")
        self.bar.refresh_panes()
        self.bar.filter_text("foobar")

    def test_restore_entry_text(self):
        self.bar.filter_text("foobar")
        self.bar.save()
        self.bar._set_text("nope")
        self.bar.restore()
        self.assertEqual(self.bar._get_text(), "foobar")
        run_gtk_loop()
        self.assertEqual(self.emit_count, 1)

    def test_restore_selection(self):
        self.bar.activate()
        self.bar.filter("artist", ["piman"])
        self.bar.save()
        self.bar.unfilter()
        self.bar.restore()
        self.bar.activate()
        run_gtk_loop()
        for song in self.last:
            assert "piman" in song.list("artist")

    def test_set_all_panes(self):
        self.bar.activate()
        self.bar.set_all_panes()

    def test_restore_pane_width(self):
        config.set("browsers", "panes", "artist\talbum")
        self.bar.set_all_panes()

        paned = self.bar.multi_paned.get_paned()
        paned.set_relative(0.8)
        self.bar.set_all_panes()
        self.assertAlmostEqual(paned.get_relative(), 0.8)

    def test_make_pane_widths_equal(self):
        config.set("browsers", "panes", "artist\talbum\t~year\t~#track")
        self.bar.set_all_panes()
        self.bar.make_pane_widths_equal()
        paneds = self.bar.multi_paned._get_paneds()

        self.assertAlmostEqual(paneds[0].get_relative(), 1.0 / 4.0)
        self.assertAlmostEqual(paneds[1].get_relative(), 1.0 / 3.0)
        self.assertAlmostEqual(paneds[2].get_relative(), 1.0 / 2.0)

    def test_column_mode(self):
        self.bar.set_all_column_mode(ColumnMode.SMALL)
        self.bar.set_all_column_mode(ColumnMode.WIDE)
        self.bar.set_all_column_mode(ColumnMode.COLUMNAR)

    def tearDown(self):
        self.bar.destroy()
        config.quit()


class TPaneConfig(TestCase):
    def test_tag(self):
        p = PaneConfig("title")
        self.assertEqual(p.title, "Title")
        self.assertEqual(p.tags, {"title"})

        self.assertEqual(p.format(SONGS[0]), [("three", "three")])
        assert str(len(ALBUM.songs)) in p.format_display(ALBUM)
        assert not p.has_markup

    def test_numeric(self):
        a_date_format = "%Y-%m-%d"
        config.set("settings", "datecolumn_timestamp_format", a_date_format)
        p = PaneConfig("~#lastplayed")
        self.assertEqual(p.title, "Last Played")
        self.assertEqual(p.tags, {"~#lastplayed"})

        zero_date = format_date(0, a_date_format)
        self.assertEqual(p.format(SONGS[0]), [(zero_date, zero_date)])
        assert not p.has_markup

    def test_tied(self):
        p = PaneConfig("~title~artist")
        self.assertEqual(p.title, "Title / Artist")
        self.assertEqual(p.tags, {"title", "artist"})

        assert p.format(SONGS[0]) == [("three", "three"), ("<boris>", "<boris>")]
        assert not p.has_markup

    def test_pattern(self):
        p = PaneConfig("<foo>")
        self.assertEqual(p.title, "Foo")
        self.assertEqual(p.tags, {"foo"})
        assert p.has_markup

    def test_condition(self):
        p = PaneConfig("<foo|a <bar>|quux>")
        self.assertEqual(p.title, "a Bar")
        self.assertEqual(p.tags, {"bar"})
        assert p.has_markup

    def test_group(self):
        p = PaneConfig(r"a\:b:<title>")
        self.assertEqual(p.title, "A:B")
        self.assertEqual(
            set(p.format_display(ALBUM).split(", ")),
            {"one", "two", "three", "four", "xxx"},
        )

        p = PaneConfig("foo:~#lastplayed")
        self.assertEqual(p.format_display(ALBUM), "0")

        p = PaneConfig("foo:title")
        self.assertEqual(
            set(p.format_display(ALBUM).split(", ")),
            {"one", "two", "three", "four", "xxx"},
        )


class TPaneEntry(TestCase):
    def test_all_have(self):
        sel = SongsEntry("foo", "foo", SONGS)
        assert not sel.all_have("artist", "one")
        assert not sel.all_have("~#mtime", 4)
        assert sel.all_have("foo", "bar")

    def test_all(self):
        entry = AllEntry()
        conf = PaneConfig("title:artist")
        assert not entry.get_count_markup(conf)
        entry.get_markup(conf)
        self.assertEqual(list(entry.songs), [])
        assert not entry.contains_text("")
        repr(entry)

    def test_unknown(self):
        entry = UnknownEntry(SONGS)
        conf = PaneConfig("title:artist")
        self.assertEqual(entry.songs, set(SONGS))
        self.assertEqual(entry.key, "")
        assert not entry.contains_text("")
        assert util.escape(SONGS[0]("artist")) in entry.get_count_markup(conf)
        entry.get_markup(conf)
        repr(entry)

    def test_songs(self):
        entry = SongsEntry("key", "key", SONGS)
        self.assertEqual(entry.key, "key")
        conf = PaneConfig("title:artist")
        assert "boris" in entry.get_count_markup(conf)
        assert entry.get_markup(conf) == "key"
        assert entry.contains_text("key")
        repr(entry)

    def test_songs_markup(self):
        entry = SongsEntry("key", "key", SONGS)
        conf = PaneConfig("<title>")
        assert entry.get_markup(conf) == "key"


class TPane(TestCase):
    def setUp(self):
        config.init()

        lib = SongLibrary()
        self.pane = Pane(lib, "artist")

    def tearDown(self):
        self.pane.destroy()
        del self.pane
        config.quit()

    def test_init(self):
        repr(self.pane)
        self.assertEqual(self.pane.tags, {"artist"})

    def test_add_remove_and_show(self):
        with realized(self.pane):
            self.pane.add(SONGS)
        with realized(self.pane):
            self.pane.remove(SONGS)
        assert not self.pane.list("arist")

    def test_matches(self):
        assert self.pane.matches(SONGS[0])
        self.pane.fill(SONGS)
        selection = self.pane.get_selection()
        selection.unselect_all()
        selection.select_path(Gtk.TreePath(3))
        assert not self.pane.matches(SONGS[1])

    def test_fill(self):
        self.pane.fill(SONGS)

    def test_fill_selection(self):
        self.pane.fill(SONGS)

        model, paths = self.pane.get_selection().get_selected_rows()
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], Gtk.TreePath((0,)))

    def test_set_selected(self):
        self.pane.fill(SONGS)

        self.pane.set_selected([])
        self.assertEqual(self.pane.get_selected(), {None})

        self.pane.set_selected([], force_any=False)
        self.assertEqual(self.pane.get_selected(), set())

        keys = self.pane.list("artist")
        self.pane.set_selected(keys)
        self.assertEqual(self.pane.get_selected(), keys)

    def test_restore_string(self):
        self.pane.fill(SONGS)
        keys = self.pane.list("artist")
        self.pane.set_selected(keys)

        to_restore = self.pane.get_restore_string()
        self.pane.remove(SONGS)
        self.pane.parse_restore_string(to_restore)
        self.pane.fill(SONGS)
        self.assertEqual(self.pane.get_selected(), keys)


class TMultiPane(TestCase):
    def setUp(self):
        config.init()

        lib = SongLibrary()
        self.p2 = Pane(lib, "artist", self)
        self.p1 = Pane(lib, "genre", self.p2)
        self.last = None
        self.count = 0

    def fill(self, songs):
        # this class is the final pane
        self.last = songs
        self.count += 1

    def test_inhibit(self):
        self.p2.inhibit()
        self.p1.fill(SONGS)
        self.p2.uninhibit()
        self.assertEqual(self.count, 0)

    def test_pipe_through(self):
        self.p1.fill(SONGS)
        self.assertEqual(self.last, set(SONGS))
        self.assertEqual(self.count, 1)

    def test_filter_first(self):
        VALUE = "J-Pop"
        self.p1.fill(SONGS)
        keys = self.p1.list("genre")
        assert VALUE in keys
        self.p1.set_selected([VALUE], force_any=False)
        assert self.last
        for song in self.last:
            self.assertEqual(song("genre"), VALUE)
        self.assertEqual(self.count, 2)

    def tearDown(self):
        self.p1.destroy()
        self.p2.destroy()
        del self.p1
        del self.p2
        config.quit()


class TPaneModel(TestCase):
    def _verify_model(self, model):
        if len(model) == 1:
            assert not isinstance(model[0][0], AllEntry)
        elif len(model) > 1:
            assert isinstance(model[0][0], AllEntry)

            for row in list(model)[1:-1]:
                assert isinstance(row[0], SongsEntry)

            self.assertTrue(isinstance(model[-1][0], SongsEntry | UnknownEntry))

    def test_add_songs(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)
        assert isinstance(m[0][0], AllEntry)
        assert isinstance(m[-1][0], UnknownEntry)
        self.assertEqual(len(m), len(SONGS) + 1 - 1)

        m.add_songs([])
        self._verify_model(m)

        m2 = PaneModel(conf)
        for song in SONGS:
            m2.add_songs([song])
            self._verify_model(m)

        self.assertEqual(len(m), len(m2))
        for e1, e2 in zip(m.itervalues(), m2.itervalues(), strict=False):
            self.assertEqual(e1.key, e2.key)

        m3 = PaneModel(conf)
        for song in reversed(SONGS):
            m3.add_songs([song])
            self._verify_model(m)

        self.assertEqual(len(m), len(m3))
        for e1, e2 in zip(m.itervalues(), m3.itervalues(), strict=False):
            self.assertEqual(e1.key, e2.key)

    def test_add_unknown_first(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs([UNKNOWN_ARTIST])
        self._verify_model(m)
        m.add_songs(SONGS)
        self._verify_model(m)

    def test_add_songs_double(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)
        self._verify_model(m)
        m.add_songs(SONGS)
        self._verify_model(m)
        self.assertEqual(len(m), len(SONGS) + 1 - 1)

    def test_get_songs(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)

        # get none
        self.assertEqual(m.get_songs([]), set())

        # get all
        self.assertEqual(len(m.get_songs([0])), len(SONGS))
        self.assertEqual(len(m.get_songs([0, 1])), len(SONGS))

        # get one
        self.assertEqual(m.get_songs([1]), {SONGS[0]})

    def test_get_keys_by_tag(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)

        assert m.get_keys_by_tag("title", ["three"]) == ["<boris>"]
        assert m.get_keys_by_tag("nope", ["foo", ""]) == [""]

        assert m.get_keys_by_tag("artist", ["piman", "foo"]) == ["piman"]

    def test_list(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)

        assert m.list("artist") == {"<boris>", "mu", "piman", ""}

        conf = PaneConfig("<artist><foo>")
        m = PaneModel(conf)
        m.add_songs(SONGS)

        assert m.list("artist") == {"<boris>", "mu", "piman"}
        assert set(m.list("foo")) == {"nope", "bar", "quux"}

    def test_get_keys(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)
        assert not m.get_keys([])
        assert m.get_keys([0, 1]) == {None, "<boris>"}

    def test_remove_songs_keep_rows(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)
        length = len(m)
        m.remove_songs(SONGS, False)
        self._verify_model(m)
        self.assertEqual(length, len(m))
        assert not m.get_songs([r.path for r in m])

    def test_remove_songs_remove_rows(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)
        length = len(m)
        m.remove_songs(SONGS, True)
        self._verify_model(m)
        self.assertNotEqual(length, len(m))
        self.assertEqual(len(m), 0)

    def test_remove_steps(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)
        for song in SONGS:
            m.remove_songs([song], True)
            self._verify_model(m)

    def test_matches(self):
        conf = PaneConfig("artist")
        m = PaneModel(conf)
        m.add_songs(SONGS)
        assert not m.matches([], SONGS[0])
        assert m.matches([0], SONGS[0])
        assert m.matches([1], SONGS[0])
        assert not m.matches([2], SONGS[0])

        m.add_songs([UNKNOWN_ARTIST])
        self._verify_model(m)
        assert m.matches([len(m) - 1], UNKNOWN_ARTIST)


class TPanedPreferences(TestCase):
    def setUp(self):
        config.init()

    def tearDown(self):
        config.quit()

    def test_editor(self):
        x = PatternEditor()
        x.headers = x.headers
        x.destroy()
        x.destroy()

    def test_button(self):
        PreferencesButton(None).destroy()

    def test_dialog(self):
        Preferences(None).destroy()
