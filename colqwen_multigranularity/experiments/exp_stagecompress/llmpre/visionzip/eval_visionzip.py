from __future__ import annotations

import sys

from colqwen_multigranularity.experiments.exp_stagecompress.llmpre.visionzip_mrl import eval_visionzip_mrl


_DROP_WITH_VALUE = {
    "--num-query-mrl-tokens",
    "--num-doc-mrl-tokens",
    "--mrl-groups",
    "--global-mrl-token-path",
}
_DROP_FLAGS = {
    "--shared-query-doc-mrl-tokens",
}
_RENAMES = {
    "--visionzip-state-path": "--visionzip-mrl-state-path",
}


def _translate_legacy_argv(argv: list[str]) -> list[str]:
    translated = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in _DROP_WITH_VALUE:
            index += 2
            continue
        if arg in _DROP_FLAGS:
            index += 1
            continue
        translated.append(_RENAMES.get(arg, arg))
        index += 1
    return translated


def main() -> None:
    old_argv = sys.argv
    try:
        sys.argv = _translate_legacy_argv(old_argv)
        eval_visionzip_mrl.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
