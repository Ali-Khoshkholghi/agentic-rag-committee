"""Import-time compatibility shims for running RAGAS in this venv.

Two unrelated breakages, both in RAGAS's transitive deps rather than our
code. ``eval/run_eval.py`` imports this module before ``ragas`` so the
fixes are in place first; it is self-contained (nothing to hand-place in
site-packages).

1. **``_lzma``** — this venv's pyenv-3.12.0 interpreter was built without
   liblzma, so ``import lzma`` (pulled in by ``datasets``, a ragas dep)
   raises ``ModuleNotFoundError: No module named '_lzma'``. The RAGAS eval
   path never touches .xz data, so we register a stub ``_lzma`` in
   ``sys.modules`` that satisfies ``lzma.py``'s import-time names; any
   real (de)compression call raises loudly. Proper fix: rebuild the
   interpreter against liblzma (``brew install xz && pyenv install
   3.12.0``), after which this shim no-ops.

2. **``langchain_community.chat_models.vertexai``** — ragas 0.4.3's
   ``ragas/llms/base.py`` still does ``from
   langchain_community.chat_models.vertexai import ChatVertexAI``, a path
   removed in the ``langchain-community`` version resolved here. RAGAS
   only uses ``ChatVertexAI`` / ``VertexAI`` for ``isinstance`` checks in
   a "supports multiple completions" list, so a stub class is harmless.
"""

from __future__ import annotations

import importlib.util
import sys
import types


def _install_lzma_shim() -> None:
    if importlib.util.find_spec("_lzma") is not None:
        return  # real extension present — nothing to do
    if "_lzma" in sys.modules:
        return

    mod = types.ModuleType("_lzma")

    class LZMAError(Exception):
        """Raised on any real LZMA use — this build has no liblzma."""

    _MSG = "the _lzma C extension is not built into this interpreter (eval shim)"

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


def _install_vertexai_shim() -> None:
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:  # if the real module is importable, leave it alone
        __import__(mod_name)
        return
    except Exception:  # noqa: BLE001
        pass

    shim = types.ModuleType(mod_name)

    class ChatVertexAI:  # noqa: D401 - stub for isinstance() only
        """Placeholder — real ChatVertexAI isn't installed in this venv."""

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = shim


_install_lzma_shim()
_install_vertexai_shim()
