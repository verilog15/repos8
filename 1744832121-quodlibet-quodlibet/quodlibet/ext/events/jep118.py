# Copyright 2005 Joe Wreschnig
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os

import quodlibet
from quodlibet import _
from quodlibet import util
from quodlibet.qltk import Icons
from quodlibet.plugins.events import EventPlugin

outfile = os.path.join(quodlibet.get_user_dir(), "jabber")
format = """\
<tune xmlns='http://jabber.org/protocol/tune'>
 <artist>%s</artist>
 <title>%s</title>
 <source>%s</source>
 <track>%d</track>
 <length>%d</length>
</tune>"""


class JEP118(EventPlugin):
    PLUGIN_ID = "JEP-118"
    PLUGIN_NAME = _("JEP-118")
    PLUGIN_DESC_MARKUP = _("Outputs a Jabber User Tunes file to %(path)s.") % {
        "path": util.monospace("~/.quodlibet/jabber")
    }
    PLUGIN_ICON = Icons.DOCUMENT_SAVE

    def plugin_on_song_started(self, song):
        if song is None:
            try:
                with open(outfile, "w", encoding="utf-8") as f:
                    f.write("<tune xmlns='http://jabber.org/protocol/tune'/>")
            except OSError:
                pass
        else:
            try:
                with open(outfile, "w", encoding="utf-8") as f:
                    f.write(
                        format
                        % (
                            util.escape(song.comma("artist")),
                            util.escape(song.comma("title")),
                            util.escape(song.comma("album")),
                            song("~#track", 0),
                            song.get("~#length", 0),
                        )
                    )
            except OSError:
                pass
