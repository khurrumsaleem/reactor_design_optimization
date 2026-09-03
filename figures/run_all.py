"""Regenerate every final figure.  Usage: <python-with-matplotlib> run_all.py"""
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig2_dynamics.py", "fig3_designs.py", "fig4_controls.py", "fig5_cpt.py",
           "fig6_steerability.py", "fig7_empirical.py", "ed1_correlation.py", "ed2_steerability_supp.py"]
fails = []
for s in SCRIPTS:
    print(f"\n=== {s}")
    if subprocess.run([sys.executable, str(HERE / s)], cwd=HERE).returncode != 0:
        fails.append(s)
print("\nFAILED:" if fails else "\nAll figures generated.", *fails)
sys.exit(1 if fails else 0)
