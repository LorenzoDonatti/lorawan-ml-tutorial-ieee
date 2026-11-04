"""Retrieve missing public inputs and enforce the recorded SHA-256 hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def request(url):
    req = urllib.request.Request(url, headers={'User-Agent':'LoRaWAN-tutorial-research/0.1'})
    return urllib.request.urlopen(req,timeout=120).read()

def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    uva=None
    for e in manifest:
        p=ROOT/e['path']
        if not p.exists():
            url=e['download_url']
            if not url:
                if uva is None:
                    uva=json.loads(request('https://dataverse.lib.virginia.edu/api/datasets/:persistentId/?persistentId=doi:10.18130/V3/RFTICK'))
                files=uva['data']['latestVersion']['files']
                matches=[f for f in files if f['dataFile']['filename']==e['original_name']]
                if len(matches)!=1: raise ValueError('Ambiguous or missing UVA file; check dataset version')
                url='https://dataverse.lib.virginia.edu/api/access/datafile/'+str(matches[0]['dataFile']['id'])
            raw=request(url)
            if hashlib.sha256(raw).hexdigest()!=e['sha256']:
                raise ValueError(f'Upstream version differs for {p.name}; review provenance before replacing it')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'], f'Hash mismatch: {p}'
        print('Verified',e['path'])

if __name__=='__main__': main()
