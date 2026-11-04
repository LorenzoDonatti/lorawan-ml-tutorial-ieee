"""Acquire immutable sources into a separate directory, checking every hash.

Offline cache reuse is explicit and never reported as a successful remote download.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,urllib.request,datetime,subprocess

ROOT=Path(__file__).resolve().parents[1]


def request(url,transport='urllib'):
    if transport=='curl':
        return subprocess.check_output(['curl','-fsSL','--connect-timeout','10','--max-time','180',url])
    req=urllib.request.Request(url,headers={'User-Agent':'LoRaWAN-Tutorial-Reproduction/0.2'})
    with urllib.request.urlopen(req,timeout=20) as r:return r.read()


def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--offline-cache',type=Path);p.add_argument('--probe-only',action='store_true');p.add_argument('--transport',choices=['urllib','curl'],default='urllib');p.add_argument('--subset',choices=['all','core','snr'],default='all');args=p.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True);manifest=json.loads((ROOT/'data/public_sources.json').read_text());records=[];uva=None
    if args.subset != 'all':
        core = {'data/raw/antwerp/antwerp_localization.csv', 'data/raw/antwerp/antwerp_gateways.csv',
                'data/raw/bonn/samples.csv', 'data/raw/bonn/gateways.csv', 'data/raw/uva/sensor01_gatewayA.parquet'}
        wanted = core if args.subset == 'core' else {'data/raw/uva/sensor01_gatewayA.parquet'}
        manifest = [entry for entry in manifest if entry['path'] in wanted]
        assert {entry['path'] for entry in manifest} == wanted
    for entry in manifest:
        target=args.destination/entry['path'];record=dict(path=entry['path'],mode='offline_cache' if args.offline_cache else 'remote',url=entry.get('url'))
        try:
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256'];record['status']='existing_hash_verified'
            else:
                if args.offline_cache:
                    source=args.offline_cache/entry['path'];raw=source.read_bytes()
                else:
                    url=entry.get('url')
                    if not url:
                        if uva is None:uva=json.loads(request('https://dataverse.lib.virginia.edu/api/datasets/:persistentId/?persistentId=doi:10.18130/V3/RFTICK',args.transport))
                        matches=[f for f in uva['data']['latestVersion']['files'] if f['dataFile']['filename']==entry['original_name']]
                        if len(matches)!=1:raise ValueError('Missing or ambiguous UVA filename')
                        ident=matches[0]['dataFile']['id'];url=f'https://dataverse.lib.virginia.edu/api/access/datafile/{ident}'
                        record.update(url=url,resolved_file_id=ident,dataset_version=uva['data']['latestVersion'].get('versionNumber'))
                    raw=request(url,args.transport)
                if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Source checksum differs; refusing replacement')
                record['status']='hash_verified'
                if not args.probe_only:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            record['bytes']=entry['bytes']
        except Exception as exc:record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        records.append(record);print(entry['path'],record['status'],flush=True)
    report=dict(subset=args.subset,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),entries=records,complete=all(r['status']!='failed' for r in records),remote_attempt=not bool(args.offline_cache),transport=args.transport,probe_only=args.probe_only)
    (args.destination/('acquisition_report.json' if args.subset == 'all' else f'acquisition_{args.subset}.json')).write_text(json.dumps(report,indent=2)+'\n')
    if not report['complete']:raise SystemExit(2)

if __name__=='__main__':main()
