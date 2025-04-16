# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from tests import mkstemp, TestCase, get_data_path

import os

from quodlibet import formats
from quodlibet import config

from shutil import copyfileobj


class TestMetaDataBase(TestCase):
    base = get_data_path("silence-44-s")

    def setUp(self):
        """Copy the base silent file to a temp name/location and load it"""

        config.init()
        fd, self.filename = mkstemp(suffix=self.ext, text=False)
        dst = os.fdopen(fd, "wb")
        src = open(self.base + self.ext, "rb")
        copyfileobj(src, dst)
        dst.close()
        self.song = formats.MusicFile(self.filename)

    def tearDown(self):
        """Delete the temp file"""

        os.remove(self.filename)
        del self.filename
        del self.song
        config.quit()


class _TestMetaDataMixin:
    def test_base_data(self):
        self.assertEqual(self.song["artist"], "piman\njzig")
        self.assertEqual(self.song["album"], "Quod Libet Test Data")
        self.assertEqual(self.song["title"], "Silence")

    def test_mutability(self):
        assert not self.song.can_change("=foo")
        assert not self.song.can_change("foo~bar")
        assert self.song.can_change("artist")
        assert self.song.can_change("title")
        assert self.song.can_change("tracknumber")
        assert self.song.can_change("somebadtag")
        assert self.song.can_change("some%punctuated:tag.")

    def _test_tag(self, tag, values, remove=True):
        assert self.song.can_change(tag)
        for value in values:
            self.song[tag] = value
            self.song.write()
            written = formats.MusicFile(self.filename)
            self.assertEqual(written[tag], value)
            if remove:
                del self.song[tag]
                self.song.write()
                deleted = formats.MusicFile(self.filename)
                assert tag not in deleted

    def test_artist(self):  # a normalish tag
        self._test_tag("artist", ["me", "you\nme", "\u6d5c\u5d0e\u3042\u3086\u307f"])

    def test_date(self):  # unusual special handling for mp3s
        self._test_tag("date", ["2004", "2005", "2005-06-12"], False)

    def test_genre(self):  # unusual special handling for mp3s
        self._test_tag(
            "genre",
            [
                "Pop",
                "Rock\nClassical",
                "Big Bird",
                "\u30a2\u30cb\u30e1\u30b5\u30f3\u30c8\u30e9",
            ],
        )

    def test_odd_performer(self):
        values = ["A Person", "Another"]
        self._test_tag("performer:vocals", values)
        self._test_tag("performer:guitar", values)

    def test_wackjob(self):  # undefined tag
        self._test_tag(
            "wackjob",
            ["Jelly\nDanish", "Muppet", "\u30cf\u30f3\u30d0\u30fc\u30ac\u30fc"],
        )


tags = [
    "album",
    "arranger",
    "artist",
    "author",
    "comment",
    "composer",
    "conductor",
    "copyright",
    "discnumber",
    "encodedby",
    "genre",
    "isrc",
    "language",
    "license",
    "lyricist",
    "organization",
    "performer",
    "title",
    "tracknumber",
    "version",
    "xyzzy_undefined_tag",
    "musicbrainz_trackid",
    "releasecountry",
]

for ext in formats.loaders.keys():
    if os.path.exists(TestMetaDataBase.base + ext):
        extra_tests = {}
        for tag in tags:
            if tag in ["artist", "date", "genre"]:
                continue

            def _test_tag(self, tag=tag):
                self._test_tag(tag, ["a"])

            extra_tests["test_tag_" + tag] = _test_tag

            def _test_tags(self, tag=tag):
                self._test_tag(tag, ["b\nc"])

            extra_tests["test_tags_" + tag] = _test_tags

        name = "MetaData" + ext
        testcase = type(name, (TestMetaDataBase, _TestMetaDataMixin), extra_tests)
        testcase.ext = ext
        globals()[name] = testcase
