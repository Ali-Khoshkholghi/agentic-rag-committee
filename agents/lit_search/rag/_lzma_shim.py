"""Import-time ``_lzma`` stub for interpreters built without liblzma.

This venv's pyenv-3.12.0 interpreter was built without liblzma, so ``import
lzma`` — pulled in transitively by ``datasets`` (a ``sentence-transformers``
dep) the moment anything imports ``SentenceTransformer`` or ``CrossEncoder``
— raises ``ModuleNotFoundError: No module named '_lzma'``. Nothing on the
retrieval path ever touches ``.xz`` data, so this registers a stub
``_lzma`` in ``sys.modules`` that satisfies ``lzma.py``'s import-time names;
any real (de)compression call raises loudly instead of silently misbehaving.

Call :func:`install` before the first ``sentence_transformers`` import.
``agents/lit_search/rag/ingest.py`` does this at module import time so
every consumer — the live agent (via ``retriever.py`` -> ``ingest.py``),
``demo.py``, ``route_demo.py`` — is covered without each one remembering
to do it. ``eval/_ragas_compat.py`` has its own independent copy of this
same fix (ragas pulls in ``datasets`` too, plus an unrelated vertexai
shim) — both are no-ops once either has already run, or once the
interpreter is rebuilt against liblzma (``brew install xz && pyenv install
3.12.0``).
"""

from __future__ import annotations

import importlib.util
import sys
import types


def install() -> None:
    if importlib.util.find_spec("_lzma") is not None:
        return  # real extension present — nothing to do
    if "_lzma" in sys.modules:
        return  # already stubbed (e.g. by eval/_ragas_compat.py)

    mod = types.ModuleType("_lzma")

    class LZMAError(Exception):
        """Raised on any real LZMA use — this build has no liblzma."""

    _MSG = "the _lzma C extension is not built into this interpreter (rag shim)"

    def _unavailable(*_a, **_kw):
        raise LZMAError(_MSG)

    # Constants lzma.py binds at import time (as default-arg values).
    consts = {
        "CHECK_NONE": 0, "CHECK_CRC32": 1, "CHECK_CRC64": 4, "CHECK_SHA256": 10,
        "CHECK_ID_MAX": 15, "CHECK_UNKNOWN": 16,
        "FILTER_LZMA1": 4611686018427387905, "FILTER_LZMA2": 33, "FILTER_DELTA": 3,
        "FILTER_X86": 4, "FILTER_IA64": 6, "FILTER_ARM": 7, "FILTER_ARMTHUMB": 8,
        "FILTER_POWERPC": 5, "FILTER_SPARC": 9,
        "FORMAT_AUTO": 0, "FORMAT_XZ": 1, "FORMAT_ALONE": 2, "FORMAT_RAW": 3,
        "MF_HC3": 3, "MF_HC4": 4, "MF_BT2": 18, "MF_BT3": 19, "MF_BT4": 20,
        "MODE_FAST": 1, "MODE_NORMAL": 2,
        "PRESET_DEFAULT": 6, "PRESET_EXTREME": 1 << 31,
    }
    for name, value in consts.items():
        setattr(mod, name, value)
    mod.LZMAError = LZMAError
    mod.LZMACompressor = _unavailable
    mod.LZMADecompressor = _unavailable
    mod._encode_filter_properties = _unavailable
    mod._decode_filter_properties = _unavailable
    mod.is_check_supported = lambda check_id: check_id in (0, 1)

    sys.modules["_lzma"] = mod
