# -*- coding: utf-8 -*-
"""Exécute la suite historique tests/smoke.py (429 vérifications de bout en
bout : formats, calibration, empilement, interface…) dans un processus
séparé, pour qu'elle fasse partie de « pytest ». Ignorer avec -m "not slow"."""
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.slow
def test_smoke_suite():
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["ASTROFORGE_HOME"] = tempfile.mkdtemp(prefix="astroforge-smoke-")
    r = subprocess.run([sys.executable, os.path.join(HERE, "smoke.py")], env=env,
                       capture_output=True, text=True, timeout=3600,
                       cwd=os.path.dirname(HERE))
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-40:])
    failed = [ln for ln in r.stdout.splitlines() if ln.startswith("✘") or ln.startswith("✖")]
    assert r.returncode == 0, f"{len(failed)} échec(s) :\n" + "\n".join(failed[:30]) + "\n" + tail
    assert "tests réussis" in r.stdout
