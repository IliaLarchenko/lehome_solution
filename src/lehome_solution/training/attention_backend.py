"""Select and install the gemma attention backend.

Only one patch can monkey-patch ``gemma.Attention``, so the two flags pick
one of four modes:

    use_flash_attention  use_xsa  →  backend
    False                False       vanilla (no patch)
    True                 False       flash_attention_patch
    False                True        xsa_attention_patch(use_flash_attention=False)
    True                 True        xsa_attention_patch(use_flash_attention=True)

Call ``install(use_flash_attention=..., use_xsa=...)`` BEFORE constructing the
model — the patch is picked up at construction time.
"""

import logging

logger = logging.getLogger(__name__)


def install(*, use_flash_attention: bool, use_xsa: bool) -> str:
    """Install the attention patch matching the flags. Returns the backend name."""
    if use_xsa:
        from lehome_solution.training import xsa_attention_patch
        xsa_attention_patch.install(use_flash_attention=use_flash_attention)
        return "xsa+flash" if use_flash_attention else "xsa"
    if use_flash_attention:
        from lehome_solution.training import flash_attention_patch
        flash_attention_patch.install()
        return "flash"
    logger.info("Using vanilla gemma.Attention (no flash, no XSA)")
    return "vanilla"
