"""支持 `python -m eval_vlm ...`。"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
