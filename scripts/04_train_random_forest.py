from __future__ import annotations

import sys

from _bootstrap import bootstrap


bootstrap()

from image_edit_detection.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))

