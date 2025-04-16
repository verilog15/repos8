# Copyright 2004-2005 Joe Wreschnig, Michael Urman
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from mutagen.musepack import Musepack

from ._audio import translate_errors
from ._apev2 import APEv2File


class MPCFile(APEv2File):
    format = "Musepack"
    mimes = ["audio/x-musepack", "audio/x-mpc"]

    def __init__(self, filename):
        with translate_errors():
            audio = Musepack(filename)

        super().__init__(filename, audio)
        self["~#length"] = audio.info.length
        self["~#bitrate"] = int(audio.info.bitrate / 1000)
        self["~#channels"] = audio.info.channels
        self["~#samplerate"] = audio.info.sample_rate

        version = audio.info.version
        self["~codec"] = f"{self.format} SV{version:d}"

        try:
            if audio.info.title_gain:
                track_g = f"{audio.info.title_gain:+0.2f} dB"
                self.setdefault("replaygain_track_gain", track_g)
            if audio.info.album_gain:
                album_g = f"{audio.info.album_gain:+0.2f} dB"
                self.setdefault("replaygain_album_gain", album_g)
            if audio.info.title_peak:
                track_p = str(audio.info.title_peak * 2)
                self.setdefault("replaygain_track_peak", track_p)
            if audio.info.album_peak:
                album_p = str(audio.info.album_peak * 2)
                self.setdefault("replaygain_album_peak", album_p)
        except AttributeError:
            pass

        self.sanitize(filename)


loader = MPCFile
types = [MPCFile]
extensions = [".mpc", ".mp+"]
