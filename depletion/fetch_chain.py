"""Download the OpenMC simplified CASL PWR depletion chain (228 nuclides) and
verify its SHA-256 against the value recorded in chain_casl_pwr.provenance.json.

Source page: https://openmc.org/data/
Usage: python fetch_chain.py            # writes depletion/chain_casl_pwr.xml
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
prov = json.loads((HERE / 'chain_casl_pwr.provenance.json').read_text())
dest = HERE / 'chain_casl_pwr.xml'

if not dest.exists():
    print(f'downloading {prov["url"]}')
    urllib.request.urlretrieve(prov['url'], dest)
digest = hashlib.sha256(dest.read_bytes()).hexdigest()
if digest != prov['sha256']:
    dest.unlink()
    sys.exit(f'checksum mismatch ({digest}); file removed, re-run fetch_chain.py')
print(f'ok: {dest} sha256={digest}')
