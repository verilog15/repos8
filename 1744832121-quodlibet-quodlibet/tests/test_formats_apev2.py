# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from tests import TestCase, get_data_path

import os
from io import BytesIO

import mutagen

from mutagen.apev2 import BINARY, APEValue

from quodlibet.formats.monkeysaudio import MonkeysAudioFile
from quodlibet.formats.mpc import MPCFile
from quodlibet.formats.wavpack import WavpackFile
from quodlibet.formats._image import APICType, EmbeddedImage

from .helper import get_temp_copy


class TAPEv2FileMixin:
    def test_can_change(self):
        self.assertEqual(self.s.can_change(), True)
        self.assertEqual(self.s.can_change("~"), False)
        self.assertEqual(self.s.can_change("a"), False)
        self.assertEqual(self.s.can_change("OggS"), True)
        self.assertEqual(self.s.can_change("\xc3\xa4\xc3\xb6"), False)
        self.assertEqual(self.s.can_change("sUbtitle"), False)
        self.assertEqual(self.s.can_change("indeX"), False)
        self.assertEqual(self.s.can_change("yEar"), False)

    def test_trans_keys(self):
        self.s["date"] = "2010"
        self.s.write()
        m = mutagen.apev2.APEv2(self.f)
        self.assertEqual(m["Year"], "2010")
        m["yEar"] = "2011"
        m.save()
        self.s.reload()
        self.assertEqual(self.s["date"], "2011")

    def test_ignore(self):
        for tag in ["inDex", "index"]:
            m = mutagen.apev2.APEv2(self.f)
            m[tag] = "foobar"
            m.save()
            self.s.reload()
            self.assertEqual(self.s.get(tag), None)
            m = mutagen.apev2.APEv2(self.f)
            self.assertEqual(m[tag], "foobar")

    def test_multi_case(self):
        self.s["AA"] = "B"
        self.s["aa"] = "C"
        self.s["BB"] = "D"
        self.s["Aa"] = "E"
        self.s.write()
        self.s.reload()
        self.assertEqual(set(self.s["aa"].split()), {"C", "B", "E"})

    def test_binary_ignore(self):
        m = mutagen.apev2.APEv2(self.f)
        m["foo"] = APEValue(b"bar", BINARY)
        m.save()
        self.s.reload()
        self.assertEqual(self.s.get("foo"), None)
        self.s.write()
        m = mutagen.apev2.APEv2(self.f)
        assert "foo" in m

    def test_titlecase(self):
        self.s["isRc"] = "1234"
        self.s["fOoBaR"] = "5678"
        self.s.write()
        self.s.reload()
        assert "isrc" in self.s
        assert "foobar" in self.s
        m = mutagen.apev2.APEv2(self.f)
        assert "ISRC" in m
        assert "Foobar" in m

    def test_disc_mapping(self):
        m = mutagen.apev2.APEv2(self.f)
        m["disc"] = "99/102"
        m.save()
        self.s.reload()
        self.assertEqual(self.s("~#disc"), 99)
        self.assertEqual(self.s("discnumber"), "99/102")

        self.s["discnumber"] = "77/88"
        self.s.write()
        m = mutagen.apev2.APEv2(self.f)
        self.assertEqual(m["disc"], "77/88")

    def test_track_mapping(self):
        m = mutagen.apev2.APEv2(self.f)
        m["track"] = "99/102"
        m.save()
        self.s.reload()
        self.assertEqual(self.s("~#track"), 99)
        self.assertEqual(self.s("tracknumber"), "99/102")

        self.s["tracknumber"] = "77/88"
        self.s.write()
        m = mutagen.apev2.APEv2(self.f)
        self.assertEqual(m["track"], "77/88")


class TMPCFileAPEv2(TestCase, TAPEv2FileMixin):
    def setUp(self):
        self.f = get_temp_copy(get_data_path("silence-44-s.mpc"))
        self.s = MPCFile(self.f)

    def tearDown(self):
        os.unlink(self.f)


class TMAFile(TestCase, TAPEv2FileMixin):
    def setUp(self):
        self.f = get_temp_copy(get_data_path("silence-44-s.ape"))
        self.s = MonkeysAudioFile(self.f)

    def tearDown(self):
        os.unlink(self.f)

    def test_format_codec(self):
        self.assertEqual(self.s("~format"), "Monkey's Audio")
        self.assertEqual(self.s("~codec"), "Monkey's Audio")
        self.assertEqual(self.s("~encoding"), "")

    def test_channels(self):
        assert self.s("~#channels") == 2

    def test_samplerate(self):
        assert self.s("~#samplerate") == 44100

    def test_bitdepth(self):
        assert self.s("~#bitdepth") == 16


def test_ma_file_old():
    s = MonkeysAudioFile(get_data_path("mac-396.ape"))

    assert s("~format") == "Monkey's Audio"
    assert s("~codec") == "Monkey's Audio"
    assert s("~encoding") == ""
    assert s("~#channels") == 2
    assert s("~#samplerate") == 44100
    # depends on the mutagen version
    assert s("~#bitdepth", 0) in (0, 16)


class TWavpackFileAPEv2(TestCase, TAPEv2FileMixin):
    def setUp(self):
        self.f = get_temp_copy(get_data_path("silence-44-s.wv"))
        self.s = WavpackFile(self.f)

    def tearDown(self):
        os.unlink(self.f)

    def test_format_codec(self):
        self.assertEqual(self.s("~format"), "WavPack")
        self.assertEqual(self.s("~codec"), "WavPack")
        self.assertEqual(self.s("~encoding"), "")


class TWvCoverArt(TestCase):
    def setUp(self):
        self.f = get_temp_copy(get_data_path("coverart.wv"))
        self.s = WavpackFile(self.f)

    def tearDown(self):
        os.unlink(self.f)

    def test_get_primary_image(self):
        cover = self.s.get_primary_image()
        assert cover
        self.assertEqual(cover.type, APICType.COVER_FRONT)

    def test_get_images(self):
        covers = self.s.get_images()
        self.assertEqual(len(covers), 2)
        types = [c.type for c in covers]
        self.assertEqual(types, [APICType.COVER_FRONT, APICType.COVER_BACK])

    def test_can_change_images(self):
        assert self.s.can_change_images

    def test_clear_images(self):
        # cover case
        image = self.s.get_primary_image()
        assert image
        self.s.clear_images()
        assert not self.s.has_images
        self.s.reload()
        image = self.s.get_primary_image()
        assert not image

        # no cover case
        self.s.clear_images()

    def test_set_image(self):
        fileobj = BytesIO(b"foo")
        image = EmbeddedImage(fileobj, "image/jpeg", 10, 10, 8)
        self.s.set_image(image)
        assert self.s.has_images

        images = self.s.get_images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mime_type, "image/")
        self.assertEqual(images[0].read(), b"foo")

    def test_set_image_no_tag(self):
        m = mutagen.apev2.APEv2(self.f)
        m.delete()
        fileobj = BytesIO(b"foo")
        image = EmbeddedImage(fileobj, "image/jpeg", 10, 10, 8)
        self.s.set_image(image)
        images = self.s.get_images()
        self.assertEqual(len(images), 1)
