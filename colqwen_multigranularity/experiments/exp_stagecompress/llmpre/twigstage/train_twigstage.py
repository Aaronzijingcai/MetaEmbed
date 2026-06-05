from __future__ import annotations

import sys

from colqwen_multigranularity.experiments.exp_stagecompress.llmpre.twigmrl import train_twigmrl


_DROP_WITH_VALUE = {
    "--num-query-mrl-tokens",
    "--num-doc-mrl-tokens",
    "--mrl-groups",
    "--global-mrl-token-path",
}
_DROP_FLAGS = {
    "--shared-query-doc-mrl-tokens",
    "--global-mrl-skip-save",
}
_RENAMES = {
    "--twigstage-state-path": "--twigmrl-state-path",
    "--twigstage-mode": "--twigmrl-mode",
    "--twigstage-exit-layer": "--twigmrl-exit-layer",
    "--twigstage-keep-ratios": "--twigmrl-keep-ratios",
    "--twigstage-temperature": "--twigmrl-temperature",
    "--twigstage-min-mask-value": "--twigmrl-min-mask-value",
    "--twigstage-train-prune": "--twigmrl-train-prune",
    "--twigstage-no-context": "--twigmrl-no-context",
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
        train_twigmrl.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
