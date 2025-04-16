# Copyright 2014 Christoph Reiter
#           2017-2020 Nick Boultbee
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import signal
import stat
from tempfile import mkstemp

try:
    import fcntl
except ImportError:
    pass

from gi.repository import GLib
from senf import fsn2bytes

from quodlibet import print_d, print_e
from quodlibet.util.path import mkdir

FIFO_TIMEOUT = 10
"""time in seconds until we give up writing/reading"""


class FIFOError(Exception):
    pass


def _sigalrm_timeout(*args):
    raise TimeoutError


def _write_fifo(fifo_path, data):
    """Writes the data to the FIFO or raises `FIFOError`"""

    assert isinstance(data, bytes)

    # This will raise if the FIFO doesn't exist or there is no reader
    try:
        fifo = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        try:
            os.unlink(fifo_path)
        except OSError:
            pass
        raise
    else:
        try:
            os.close(fifo)
        except OSError:
            pass

    try:
        signal.signal(signal.SIGALRM, _sigalrm_timeout)
        signal.alarm(FIFO_TIMEOUT)
        with open(fifo_path, "wb") as f:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            f.write(data)
    except (OSError, TimeoutError) as e:
        # Unable to write to the fifo. Removing it.
        try:
            os.unlink(fifo_path)
        except OSError:
            pass
        raise FIFOError(f"Couldn't write to fifo {fifo_path!r}") from e


def split_message(data):
    """Split incoming data in pairs of (command, FIFO path or `None`)

    This supports two data formats:
    Newline-separated commands without a return FIFO path.
    and "NULL<command>NULL<fifo-path>NULL"

    Args:
        data (bytes)
    Returns:
        Tuple[bytes, bytes]
    Raises:
        ValueError
    """

    assert isinstance(data, bytes)

    while data:
        elem, null, data = data.partition(b"\x00")
        if elem:  # newline-separated commands
            yield from ((line, None) for line in elem.splitlines())
            data = null + data
        else:  # NULL<command>NULL<fifo-path>NULL
            cmd, fifo_path, data = data.split(b"\x00", 2)
            yield (cmd, fifo_path)


def write_fifo(fifo_path, data):
    """Writes the data to the FIFO and returns a response.

    Args:
        fifo_path (pathlike)
        data (bytes)
    Returns:
        bytes
    Raises:
        FIFOError
    """

    assert isinstance(data, bytes)

    fd, filename = mkstemp()
    try:
        os.close(fd)
        os.unlink(filename)
        # mkfifo fails if the file exists, so this is safe.
        os.mkfifo(filename, 0o600)

        _write_fifo(
            fifo_path, b"\x00".join([b"", data, fsn2bytes(filename, None), b""])
        )

        try:
            signal.signal(signal.SIGALRM, _sigalrm_timeout)
            signal.alarm(FIFO_TIMEOUT)
            with open(filename, "rb") as h:
                signal.signal(signal.SIGALRM, signal.SIG_IGN)
                return h.read()
        except TimeoutError as e:
            # In case the main instance deadlocks we can write to it, but
            # reading will time out. Assume it is broken and delete the
            # fifo.
            try:
                os.unlink(fifo_path)
            except OSError:
                pass
            raise FIFOError("Timeout reached") from e
    except OSError as e:
        raise FIFOError(*e.args) from e
    finally:
        try:
            os.unlink(filename)
        except OSError:
            pass


def fifo_exists(fifo_path):
    """Returns whether a FIFO exists (and is writeable).

    Args:
        fifo_path (pathlike)
    Returns:
        bool
    """

    # https://github.com/quodlibet/quodlibet/issues/1131
    # FIXME: There is a race where control() creates a new file
    # instead of writing to the FIFO, confusing the next QL instance.
    # Remove non-FIFOs here for now.
    try:
        if not stat.S_ISFIFO(os.stat(fifo_path).st_mode):
            print_d(f"{fifo_path!r} not a FIFO. Removing it.")
            os.remove(fifo_path)
    except OSError:
        pass
    return os.path.exists(fifo_path)


class FIFO:
    """Creates and reads from a FIFO"""

    _source_id = None

    def __init__(self, path, callback):
        """
        Args:
            path (pathlike)
            callback (Callable[[bytes], None])
        """

        self._callback = callback
        self._path = path

    def destroy(self):
        """After destroy() the callback will no longer be called
        and the FIFO can no longer be used. Can be called multiple
        times.
        """

        if self._source_id is not None:
            GLib.source_remove(self._source_id)
            self._source_id = None

        try:
            os.unlink(self._path)
        except OSError:
            pass

    def open(self, ignore_lock=False):
        """Create the FIFO and listen to it.

        Raises:
            FIFOError in case another process is already using it.
        """
        from quodlibet import qltk

        mkdir(os.path.dirname(self._path))
        try:
            os.mkfifo(self._path, 0o600)
        except OSError:
            # maybe exists, we'll fail below otherwise
            pass

        try:
            fifo = os.open(self._path, os.O_NONBLOCK)
        except OSError:
            return

        while True:
            try:
                fcntl.flock(fifo, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except InterruptedError:  # EINTR
                continue
            except BlockingIOError as e:  # EWOULDBLOCK
                if not ignore_lock:
                    raise FIFOError("fifo already locked") from e
            except OSError as e:
                print_d(f"fifo locking failed: {e!r}")
            break

        try:
            f = os.fdopen(fifo, "rb", 4096)
        except OSError as e:
            print_e(f"Couldn't open FIFO ({e!r})")
        else:
            self._source_id = qltk.io_add_watch(
                f,
                GLib.PRIORITY_DEFAULT,
                GLib.IO_IN | GLib.IO_ERR | GLib.IO_HUP,
                self._process,
            )

    def _process(self, source, condition):
        if condition in {GLib.IO_ERR, GLib.IO_HUP}:
            self.open(ignore_lock=True)
            return False

        while True:
            try:
                data = source.read()
            except InterruptedError:  # EINTR
                continue
            except (BlockingIOError, PermissionError):  # EWOULDBLOCK, EACCES
                return True
            except OSError:
                self.open(ignore_lock=True)
                return False
            break

        if not data:
            self.open(ignore_lock=True)
            return False

        self._callback(data)

        return True
