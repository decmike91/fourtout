# -*- coding: utf-8 -*-
"""python -m astroforge … : ligne de commande (sans argument : interface)."""
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from main import main as gui_main
        sys.exit(gui_main())
    from astroforge.cli import main
    sys.exit(main())
