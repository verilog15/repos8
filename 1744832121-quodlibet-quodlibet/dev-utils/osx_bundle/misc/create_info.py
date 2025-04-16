#!/usr/bin/env python3

import sys
import plistlib


if __name__ == "__main__":
    assert sys.version_info[0] == 3

    # "quodlibet" "3.4.0"
    app_id, app_version = sys.argv[1:]
    app_id_lower = app_id.lower()

    plist = dict(
        CFBundleExecutable=app_id_lower,
        CFBundleIconFile="%s.icns" % app_id_lower,
        CFBundleIdentifier="io.github.quodlibet.%s" % app_id,
        CFBundleInfoDictionaryVersion="6.0",
        CFBundlePackageType="APPL",
        CFBundleShortVersionString=app_version,
        CFBundleVersion=app_version,
        LSMinimumSystemVersion="10.9",
        AppleMagnifiedMode=False,
        NSHighResolutionCapable=True,
        CFBundleSignature="????",
    )

    if app_id_lower == "exfalso":
        plist.update(dict(
            CFBundleName="Ex Falso",
            CFBundleDocumentTypes=[
                dict(
                    CFBundleTypeOSTypes=["fold"],
                    CFBundleTypeRole="Viewer",
                ),
            ],
        ))
    elif app_id_lower == "quodlibet":
        plist.update(dict(
            CFBundleName="Quod Libet",
            CFBundleDocumentTypes=[
                dict(
                    CFBundleTypeExtensions=[
                        "3g2", "3gp", "3gp2", "669", "aac", "adif",
                        "adts", "aif", "aifc", "aiff", "amf", "ams",
                        "ape", "asf", "dsf", "dsm", "far", "flac", "gdm",
                        "it", "m4a", "m4b", "m4v", "med", "mid", "mod", "mp+",
                        "mp1", "mp2", "mp3", "mp4", "mpc", "mpeg",
                        "mpg", "mt2", "mtm", "oga", "ogg", "oggflac",
                        "ogv", "okt", "opus", "s3m", "spc", "spx",
                        "stm", "tta", "ult", "vgm", "wav", "wma",
                        "wmv", "wv", "xm",
                    ],
                    CFBundleTypeRole="Viewer",
                    CFBundleTypeIconFile="quodlibet.icns",
                ),
            ],
        ))
    else:
        assert 0, app_id_lower

    print(plistlib.dumps(plist).decode("utf-8"))



