import re
import sys
import locale
import io

def re_findall(pattern, string):
    return [m.groupdict() for m in re.finditer(pattern, string)]


def jaccard(x1, x2, y1, y2):
    # Calculate jaccard index
    intersection = max(0, min(x2, y2)-max(x1, y1))
    filled_union = max(x2, y2) - min(x1, y1)
    return intersection/filled_union if filled_union > 0 else 0


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default


def _windows_write_string(s, out, skip_errors=True):
    """ Returns True if the string was written using special methods,
    False if it has yet to be written out."""
    # Adapted from http://stackoverflow.com/a/3259271/35070

    import ctypes
    import ctypes.wintypes

    WIN_OUTPUT_IDS = {
        1: -11,
        2: -12,
    }

    try:
        fileno = out.fileno()
    except AttributeError:
        # If the output stream doesn't have a fileno, it's virtual
        return False
    except io.UnsupportedOperation:
        # Some strange Windows pseudo files?
        return False
    if fileno not in WIN_OUTPUT_IDS:
        return False

    GetStdHandle = ctypes.WINFUNCTYPE(
        ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD)(
        ('GetStdHandle', ctypes.windll.kernel32))
    h = GetStdHandle(WIN_OUTPUT_IDS[fileno])

    WriteConsoleW = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HANDLE, ctypes.wintypes.LPWSTR,
        ctypes.wintypes.DWORD, ctypes.POINTER(ctypes.wintypes.DWORD),
        ctypes.wintypes.LPVOID)(('WriteConsoleW', ctypes.windll.kernel32))
    written = ctypes.wintypes.DWORD(0)

    GetFileType = ctypes.WINFUNCTYPE(ctypes.wintypes.DWORD, ctypes.wintypes.DWORD)(
        ('GetFileType', ctypes.windll.kernel32))
    FILE_TYPE_CHAR = 0x0002
    FILE_TYPE_REMOTE = 0x8000
    GetConsoleMode = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HANDLE,
        ctypes.POINTER(ctypes.wintypes.DWORD))(
        ('GetConsoleMode', ctypes.windll.kernel32))
    INVALID_HANDLE_VALUE = ctypes.wintypes.DWORD(-1).value

    def not_a_console(handle):
        if handle == INVALID_HANDLE_VALUE or handle is None:
            return True
        return ((GetFileType(handle) & ~FILE_TYPE_REMOTE) != FILE_TYPE_CHAR or GetConsoleMode(handle, ctypes.byref(ctypes.wintypes.DWORD())) == 0)

    if not_a_console(h):
        return False

    def next_nonbmp_pos(s):
        try:
            return next(i for i, c in enumerate(s) if ord(c) > 0xffff)
        except StopIteration:
            return len(s)

    while s:
        count = min(next_nonbmp_pos(s), 1024)

        ret = WriteConsoleW(
            h, s, count if count else 2, ctypes.byref(written), None)
        if ret == 0:
            if skip_errors:
                continue
            else:
                raise OSError('Failed to write string')
        if not count:  # We just wrote a non-BMP character
            assert written.value == 2
            s = s[1:]
        else:
            assert written.value > 0
            s = s[written.value:]
    return True

def preferredencoding():
    """Get preferred encoding.
    Returns the best encoding scheme for the system, based on
    locale.getpreferredencoding() and some further tweaks.
    """
    try:
        pref = locale.getpreferredencoding()
        'TEST'.encode(pref)
    except Exception:
        pref = 'utf-8'

    return pref

def safe_print(*objects, sep=' ', end='\n', out=None, encoding=None, flush=False):
    """
    Ensure printing to standard output can be done safely (especially on Windows).
    There are usually issues with printing emojis and non utf-8 characters.
    """

    output_string = sep.join(map(lambda x: str(x), objects)) + end

    if out is None:
        out = sys.stdout

    if sys.platform == 'win32' and encoding is None and hasattr(out, 'fileno'):
        if _windows_write_string(output_string, out):
            return

    if 'b' in getattr(out, 'mode', '') or not hasattr(out, 'buffer'):
        out.write(output_string)
    else:
        enc = encoding or getattr(out, 'encoding', None) or preferredencoding()
        byt = output_string.encode(enc, 'ignore')
        out.buffer.write(byt)

    if flush and hasattr(out, 'flush'):
        out.flush()