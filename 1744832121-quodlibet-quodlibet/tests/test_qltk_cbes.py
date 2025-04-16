# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from tests import TestCase, mkstemp

import os

from quodlibet.qltk.cbes import ComboBoxEntrySave, StandaloneEditor
import quodlibet.config


class TComboBoxEntrySave(TestCase):
    memory = "pattern 1\npattern 2\n"
    saved = "pattern text\npattern name\n"

    def setUp(self):
        quodlibet.config.init()
        h, self.fname = mkstemp()
        os.close(h)
        with open(self.fname, "w") as f:
            f.write(self.memory)

        with open(self.fname + ".saved", "w") as f:
            f.write(self.saved)
        self.cbes = ComboBoxEntrySave(self.fname, count=2)
        self.cbes2 = ComboBoxEntrySave(self.fname, count=2)

    def test_equivalence(self):
        model1 = self.cbes.get_model()
        model2 = self.cbes2.get_model()
        self.assertEqual(model1, model2)

        rows1 = list(model1)
        rows2 = list(model2)

        for row1, row2 in zip(rows1, rows2, strict=False):
            self.assertEqual(row1[0], row2[0])
            self.assertEqual(row1[1], row2[1])
            self.assertEqual(row1[2], row2[2])

    def test_text_changed_signal(self):
        called = [0]

        def cb(*args):
            called[0] += 1

        def get_count():
            c = called[0]
            called[0] = 0
            return c

        self.cbes.connect("text-changed", cb)
        entry = self.cbes.get_child()
        entry.set_text("foo")
        self.assertEqual(get_count(), 1)
        self.cbes.prepend_text("bar")
        # in case the model got changed but the entry is still the same
        # the text-changed signal should not be triggered
        self.assertEqual(entry.get_text(), "foo")
        self.assertEqual(get_count(), 0)

    def test_shared_model(self):
        self.cbes.prepend_text("a test")
        self.test_equivalence()

    def test_initial_size(self):
        # 1 saved, Edit, separator, 2 remembered
        self.assertEqual(len(self.cbes.get_model()), 5)

    def test_prepend_text(self):
        self.cbes.prepend_text("pattern 3")
        self.memory = "pattern 3\npattern 1\n"
        self.test_save()

    def test_save(self):
        self.cbes.write()
        self.assertEqual(self.memory, open(self.fname).read())
        self.assertEqual(self.saved, open(self.fname + ".saved").read())

    def test_set_text_then_prepend(self):
        self.cbes.get_child().set_text("foobar")
        self.cbes.prepend_text("foobar")
        self.memory = "foobar\npattern 1\n"
        self.test_save()

    def tearDown(self):
        self.cbes.destroy()
        self.cbes2.destroy()
        os.unlink(self.fname)
        os.unlink(self.fname + ".saved")
        quodlibet.config.quit()


class TStandaloneEditor(TestCase):
    TEST_KV_DATA = [("Search Foo", "https://foo.com/search?q=<artist>-<title>")]

    def setUp(self):
        quodlibet.config.init()
        h, self.fname = mkstemp()
        os.close(h)
        with open(self.fname + ".saved", "w") as f:
            f.write(f"{self.TEST_KV_DATA[0][1]}\n{self.TEST_KV_DATA[0][0]}\n")
        self.sae = StandaloneEditor(self.fname, "test", None, None)

    def test_constructor(self):
        assert self.sae.model
        data = [(row[1], row[0]) for row in self.sae.model]
        self.assertEqual(data, self.TEST_KV_DATA)

    def test_load_values(self):
        values = StandaloneEditor.load_values(self.fname + ".saved")
        self.assertEqual(self.TEST_KV_DATA, values)

    def test_defaults(self):
        defaults = [("Dot-com Dream", "http://<artist>.com")]
        try:
            os.unlink(self.fname)
        except OSError:
            pass
        # Now create a new SAE without saved results and use defaults
        self.fname = "foo"
        self.sae.destroy()
        self.sae = StandaloneEditor(self.fname, "test2", defaults, None)
        self.sae.write()
        data = [(row[1], row[0]) for row in self.sae.model]
        self.assertEqual(defaults, data)

    def tearDown(self):
        self.sae.destroy()
        try:
            os.unlink(self.fname)
            os.unlink(self.fname + ".saved")
        except OSError:
            pass
        quodlibet.config.quit()
