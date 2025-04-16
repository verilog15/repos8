# Copyright 2012 Christoph Reiter
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import sys

from senf import fsnative

from tests import TestCase, get_data_path
from .helper import capture_output

from quodlibet import formats
from quodlibet.formats import (
    AudioFile,
    load_audio_files,
    dump_audio_files,
    SerializationError,
)
from quodlibet.util.picklehelper import pickle_dumps
from quodlibet import config


class TFormats(TestCase):
    def setUp(self):
        config.init()

    def tearDown(self):
        config.quit()

    def test_presence(self):
        assert formats.aac
        assert formats.aiff
        assert formats.midi
        assert formats.mod
        assert formats.monkeysaudio
        assert formats.mp3
        assert formats.mp4
        assert formats.mpc
        assert formats.spc
        assert formats.trueaudio
        assert formats.vgm
        assert formats.wav
        assert formats.wavpack
        assert formats.wma
        assert formats.xiph

    def test_loaders(self):
        assert formats.loaders[".mp3"] is formats.mp3.MP3File

    def test_migration(self):
        assert formats.mp3 is sys.modules["quodlibet.formats.mp3"]
        assert formats.mp3 is sys.modules["quodlibet/formats/mp3"]
        assert formats.mp3 is sys.modules["formats.mp3"]

        assert formats.xiph is sys.modules["formats.flac"]
        assert formats.xiph is sys.modules["formats.oggvorbis"]

    def test_filter(self):
        assert formats.filter("foo.mp3")
        assert not formats.filter("foo.doc")
        assert not formats.filter("foomp3")

    def test_music_file(self):
        path = get_data_path("silence-44-s.mp3")
        assert formats.MusicFile(path)

        # non existing
        with capture_output() as (stdout, stderr):
            song = formats.MusicFile(get_data_path("nope.mp3"))
            assert not song
            assert stderr.getvalue()

        # unknown extension
        with capture_output() as (stdout, stderr):
            song = formats.MusicFile(get_data_path("nope.xxx"))
            assert not song
            assert not stderr.getvalue()


class TPickle(TestCase):
    def setUp(self):
        types = formats.types
        instances = []
        for t in types:
            i = AudioFile.__new__(t)
            # we want to pickle/unpickle everything, since historically
            # these things ended up in the file
            dict.__init__(
                i, {b"foo": "bar", "quux": b"baz", "a": "b", "b": 42, "c": 0.25}
            )
            instances.append(i)
        self.instances = instances

    def test_load_audio_files(self):
        for protocol in [0, 1, 2]:
            data = pickle_dumps(self.instances, protocol)
            items = load_audio_files(data)
            assert len(items) == len(formats.types)
            assert all(isinstance(i, AudioFile) for i in items)

    def test_sanitized_py3(self):
        i = AudioFile.__new__(list(formats.types)[0])
        # this is something that old py2 versions could pickle
        dict.__init__(
            i,
            {
                b"bytes": b"bytes",
                "unicode": "unicode",
                b"~filename": b"somefile",
                "~mountpoint": "somemount",
                "int": 42,
                b"float": 1.25,
            },
        )
        data = pickle_dumps([i], 1)
        items = load_audio_files(data, process=True)
        i = items[0]

        assert i["bytes"] == "bytes"
        assert i["unicode"] == "unicode"
        assert isinstance(i["~filename"], fsnative)
        assert isinstance(i["~mountpoint"], fsnative)
        assert i["int"] == 42
        assert i["float"] == 1.25

    def test_dump_audio_files(self):
        data = dump_audio_files(self.instances, process=False)
        items = load_audio_files(data, process=False)

        assert len(items) == len(self.instances)
        for a, b in zip(items, self.instances, strict=False):
            a = dict(a)
            b = dict(b)
            for key in a:
                assert b[key] == a[key]
            for key in b:
                assert b[key] == a[key]

    def test_save_ascii_keys_as_bytes_on_py3(self):
        i = AudioFile.__new__(list(formats.types)[0])
        dict.__setitem__(i, "foo", "bar")
        data = dump_audio_files([i], process=True)

        items = load_audio_files(data, process=False)
        assert isinstance(list(items[0].keys())[0], bytes)

    def test_dump_empty(self):
        data = dump_audio_files([])
        assert load_audio_files(data) == []

    def test_load_audio_files_missing_class(self):
        for protocol in [0, 1, 2]:
            data = pickle_dumps(self.instances, protocol)

            items = load_audio_files(data)
            self.assertEqual(len(items), len(formats.types))
            assert all(isinstance(i, AudioFile) for i in items)

            broken = data.replace(b"SPCFile", b"FooFile")
            items = load_audio_files(broken)
            self.assertEqual(len(items), len(formats.types) - 1)
            assert all(isinstance(i, AudioFile) for i in items)

            broken = data.replace(b"formats.spc", b"formats.foo")
            items = load_audio_files(broken)
            self.assertEqual(len(items), len(formats.types) - 1)
            assert all(isinstance(i, AudioFile) for i in items)

    def test_unpickle_random_class(self):
        for protocol in [0, 1, 2]:
            data = pickle_dumps([42], protocol)
            with self.assertRaises(SerializationError):
                load_audio_files(data)
