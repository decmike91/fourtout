#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AstroForge — point d'entrée : python main.py (interface graphique).
Ligne de commande : python -m astroforge --help"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astroforge.gui.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
