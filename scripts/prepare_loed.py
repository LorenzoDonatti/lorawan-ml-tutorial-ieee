"""Import the public six-file LoED sample and form reception profiles."""
from pathlib import Path
from zipfile import ZipFile
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['log_count','rssi_mean','rssi_std','snr_mean','snr_std','sf_mean',
          'sf_std','log_median_gap','log_gap_iqr','frequency_count']

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--record',required=True);a=p.parse_args()
    archive=Path(a.archive);record=json.loads(Path(a.record).read_text())
    if hashlib.md5(archive.read_bytes()).hexdigest()!='7370baddc9b546354c26a71cd38981bc':raise ValueError('Unexpected LoED sample version')
    out=ROOT/'data/raw/loed';out.mkdir(exist_ok=True)
    hashes={};frames=[]
    with ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            if not member.endswith('.csv') or member.startswith('__'):continue
            content=z.read(member);path=out/Path(member).name
            if path.exists() and path.read_bytes()!=content:raise ValueError('Preserve existing data')
            path.write_bytes(content);hashes[str(path.relative_to(ROOT))]=hashlib.sha256(content).hexdigest()
            d=pd.read_csv(path,dtype={'device_address':str,'gateway':str,'mtype':str});d['source_file']=path.name;frames.append(d)
    d=pd.concat(frames,ignore_index=True);raw=len(d)
    d=d[d.crc_status.eq(1)&d.mtype.isin(['010','100'])&d.device_address.notna()].copy()
    d['time']=pd.to_datetime(d.time,utc=True);d['day']=d.time.dt.strftime('%Y-%m-%d')
    d=d.sort_values('time');profiles=[]
    for (day,gateway,address),g in d.groupby(['day','gateway','device_address'],sort=True):
        if len(g)<20:continue
        gaps=g.time.diff().dt.total_seconds().dropna().to_numpy()
        row=dict(day=day,gateway=gateway,address=address,row_id=f'{day}|{gateway}|{address}',
            log_count=np.log1p(len(g)),rssi_mean=g.rssi.mean(),rssi_std=g.rssi.std(ddof=0),
            snr_mean=g.snr.mean(),snr_std=g.snr.std(ddof=0),sf_mean=g.spreading_factor.mean(),
            sf_std=g.spreading_factor.std(ddof=0),log_median_gap=np.log1p(np.median(gaps)),
            log_gap_iqr=np.log1p(np.quantile(gaps,.75)-np.quantile(gaps,.25)),frequency_count=g.frequency.nunique())
        profiles.append(row)
    profiles=pd.DataFrame(profiles);assert np.isfinite(profiles[FEATURES]).all().all()
    processed=ROOT/'data/processed';processed.mkdir(exist_ok=True)
    path=processed/'loed_profiles.csv';profiles.to_csv(path,index=False)
    manifest=dict(source='https://doi.org/10.5281/zenodo.4121430',download_url='https://zenodo.org/records/4121430/files/LoED_LoRaWAN_at_edge_dataset-SAMPLE.zip?download=1',
        sample_md5=hashlib.md5(archive.read_bytes()).hexdigest(),sample_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        license=record['metadata'].get('license'),raw_rows=raw,eligible_receptions=len(d),
        files=len(frames),utc_dates=sorted(d.day.unique().tolist()),profiles=len(profiles),
        addresses=profiles.address.nunique(),features=FEATURES,hashes=hashes,
        profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        interpretation='Per UTC day, gateway and observed address; >=20 CRC-valid data-uplink receptions. No inferred device identity, transmission denominator or incident label.')
    (out/'zenodo_record.json').write_text(json.dumps(record,indent=2)+'\n')
    (ROOT/'data/loed_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['hashes','features']},indent=2))

if __name__=='__main__':main()
