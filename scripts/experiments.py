"""Small, auditable LoRaWAN studies. The protocol is docs/PROTOCOLO_PILOTO.md."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'

def geo_xy(lat, lon, origin=(50., 7.)):
    """Fixed equirectangular coordinates, only for features and grid assignment."""
    return np.column_stack([6371000*np.deg2rad(np.asarray(lon)-origin[1])*np.cos(np.deg2rad(origin[0])),
                            6371000*np.deg2rad(np.asarray(lat)-origin[0])])

def haversine(lat1,lon1,lat2,lon2):
    a,b,c,d=map(np.deg2rad,[lat1,lon1,lat2,lon2])
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*6371000*np.arcsin(np.sqrt(np.clip(h,0,1)))

def cells(lat,lon,origin,side=500):
    ij=np.floor(geo_xy(lat,lon,origin)/side).astype(int)
    return np.array([f'{i}:{j}' for i,j in ij])

def metrics(y,p):
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))))

def forest(seed=42,trees=80):
    return RandomForestRegressor(n_estimators=trees,min_samples_leaf=5,random_state=seed,n_jobs=2)

def grouped_split(groups,seed,test_size=.2):
    return next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(np.zeros(len(groups)),groups=groups))

def load_localization():
    d=pd.read_csv(RAW/'antwerp/antwerp_localization.csv')
    g=pd.read_csv(RAW/'antwerp/antwerp_gateways.csv')
    features=[c for c in d if c.startswith('rssi_')]
    return d,g,features

def run_localization(out):
    d,g,features=load_localization()
    X=d[features].fillna(-140).to_numpy()
    Y=geo_xy(d.lat,d.lon,(51.22,4.40))
    grid=cells(d.lat,d.lon,(51.22,4.40))
    chrono=[np.flatnonzero(d.split.eq(s)) for s in ['treino','validacao','teste']]
    # Be explicit about the source's Portuguese label spelling.
    if not len(chrono[1]):
        chrono[1]=np.flatnonzero(d.split.eq('validação'))
    assert all(len(x)>0 for x in chrono), d.split.value_counts()
    trva,te=grouped_split(grid,42,.15)
    tr0,va0=grouped_split(grid[trva],43,.15/.85)
    protocols={'chronological':chrono,'spatial_500m':[trva[tr0],trva[va0],te]}
    rows=[]; predictions=[]; partitions=[]; selections=[]
    def latlon(xy):
        return np.column_stack([51.22+np.rad2deg(xy[:,1]/6371000),
            4.40+np.rad2deg(xy[:,0]/(6371000*np.cos(np.deg2rad(51.22))))])
    def error(idx,xy):
        ll=latlon(xy)
        return haversine(d.lat.iloc[idx].to_numpy(),d.lon.iloc[idx].to_numpy(),ll[:,0],ll[:,1])
    gw=g.set_index('feature').loc[features]
    gwxy=geo_xy(gw.lat,gw.lon,(51.22,4.40))
    for name,(tr,va,te) in protocols.items():
        for part,idx in [('train',tr),('validation',va),('test',te)]:
            partitions.extend(dict(protocol=name,row_id=int(i),split=part,group=grid[i],timestamp=d.timestamp.iloc[i]) for i in idx)
        candidates=[]
        for k in [3,9,21]:
            model=KNeighborsRegressor(n_neighbors=k,weights='distance',n_jobs=2).fit(X[tr],Y[tr])
            candidates.append((error(va,model.predict(X[va])).mean(),k,model))
        best=min(candidates,key=lambda z:z[0]);knn=best[2]
        selections.append(dict(protocol=name,knn_k=best[1],validation_mean_error_m=best[0]))
        rf=forest(trees=100).fit(X[tr],Y[tr])
        weights=np.where(np.isnan(d[features].to_numpy()[te]),0,10**(X[te]/10))
        centroid=weights@gwxy/weights.sum(axis=1,keepdims=True)
        outputs={'median_position':np.tile(np.median(Y[tr],axis=0),(len(te),1)),
            'weighted_centroid':centroid,'knn':knn.predict(X[te]),'random_forest':rf.predict(X[te])}
        # True geographic distance to the nearest training position, in radians.
        nbr=NearestNeighbors(n_neighbors=1,metric='haversine',algorithm='ball_tree').fit(np.deg2rad(d[['lat','lon']].to_numpy()[tr]))
        nearest=nbr.kneighbors(np.deg2rad(d[['lat','lon']].to_numpy()[te]))[0][:,0]*6371000
        for model,xy in outputs.items():
            errors=error(te,xy);ll=latlon(xy)
            rows.append(dict(protocol=name,model=model,n_train=len(tr),n_validation=len(va),n_test=len(te),
                mean_error_m=errors.mean(),median_error_m=np.median(errors),p90_error_m=np.quantile(errors,.9)))
            predictions.extend(dict(protocol=name,model=model,row_id=int(i),error_m=float(e),
                true_lat=d.lat.iloc[i],true_lon=d.lon.iloc[i],pred_lat=p[0],pred_lon=p[1],
                n_gateways=int(d[features].iloc[i].notna().sum()),nearest_train_m=float(dist))
                for i,e,p,dist in zip(te,errors,ll,nearest))
    pd.DataFrame(rows).to_csv(out/'localization_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'localization_predictions.csv',index=False)
    pd.DataFrame(partitions).to_csv(out/'localization_splits.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'localization_selection.csv',index=False)
    print('Localization complete',flush=True)

def load_pathloss():
    d=pd.read_csv(RAW/'bonn/samples.csv').reset_index(names='row_id')
    g=pd.read_csv(RAW/'bonn/gateways.csv').rename(columns={'id':'gw','latitude':'gw_lat','longitude':'gw_lon'})
    d=d.merge(g,on='gw',validate='many_to_one')
    d['distance_m']=haversine(d.latitude.to_numpy(),d.longitude.to_numpy(),d.gw_lat.to_numpy(),d.gw_lon.to_numpy())
    d=d.loc[d.distance_m.gt(0)].sample(n=min(30000,len(d)),random_state=20261021).sort_values('row_id').reset_index(drop=True)
    d['log_distance']=np.log10(d.distance_m/1000)
    d['target']=15-(d.rssi+np.minimum(d.snr,0))
    d['packet_group']=d[['transceived_at','pkt_number','latitude','longitude']].astype(str).agg('|'.join,axis=1)
    d['spatial_group']=cells(d.latitude,d.longitude,(50.,7.))
    return d,['log_distance','latitude','longitude','gw_lat','gw_lon','antenna_height']

def run_pathloss(out):
    d,features=load_pathloss();X=d[features].to_numpy();y=d.target.to_numpy()
    records=[];preds=[];splits=[];audit=[]
    for seed in [17,42,73]:
        protocols={'random_receptions':train_test_split(np.arange(len(d)),test_size=.2,random_state=seed),
            'packet_group':grouped_split(d.packet_group.to_numpy(),seed),
            'spatial_500m':grouped_split(d.spatial_group.to_numpy(),seed)}
        for name,(tr,te) in protocols.items():
            for label,idx in [('train',tr),('test',te)]:
                splits.extend(dict(protocol=name,seed=seed,row_id=int(d.row_id.iloc[i]),split=label,
                    packet_group=d.packet_group.iloc[i],spatial_group=d.spatial_group.iloc[i]) for i in idx)
            base=LinearRegression().fit(X[tr,:1],y[tr])
            pred={'log_distance':base.predict(X[te,:1]),
                'ridge_geometry':make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(X[tr],y[tr]).predict(X[te]),
                'random_forest':forest(seed).fit(X[tr],y[tr]).predict(X[te]),
                'residual_forest':base.predict(X[te,:1])+forest(seed).fit(X[tr],y[tr]-base.predict(X[tr,:1])).predict(X[te])}
            audit.append(dict(protocol=name,seed=seed,n_train=len(tr),n_test=len(te),
                test_packet_seen_fraction=float(d.packet_group.iloc[te].isin(d.packet_group.iloc[tr]).mean()),
                test_cell_seen_fraction=float(d.spatial_group.iloc[te].isin(d.spatial_group.iloc[tr]).mean()),
                logdistance_intercept=float(base.intercept_),logdistance_slope=float(base.coef_[0])))
            for model,p in pred.items():
                records.append(dict(protocol=name,seed=seed,model=model,n_test=len(te),**metrics(y[te],p)))
                preds.extend(dict(protocol=name,seed=seed,model=model,row_id=int(d.row_id.iloc[i]),truth=float(t),prediction=float(v)) for i,t,v in zip(te,y[te],p))
            print('Path loss',name,seed,flush=True)
    pd.DataFrame(records).to_csv(out/'pathloss_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'pathloss_predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(out/'pathloss_splits.csv.gz',index=False)
    pd.DataFrame(audit).to_csv(out/'pathloss_audit.csv',index=False)

def hourly_link(sensor):
    d=pd.read_parquet(RAW/f'uva/{sensor}_gatewayA.parquet')
    d=d.set_index('Timestamp').sort_index()
    h=d[['RSSI (dBm)','SNR (dB)']].rename(columns={'RSSI (dBm)':'rssi','SNR (dB)':'snr'}).resample('1h').mean()
    h['n_received']=d['RSSI (dBm)'].resample('1h').count()
    # Label each aggregate by the instant at which its entire hour has closed.
    h.index=h.index+pd.Timedelta(hours=1)
    h.index.name='available_at'
    return h

def forecast_table(h,target,horizon):
    data={f'{v}_lag{k}':h[v].shift(k) for v in ['rssi','snr'] for k in range(24)}
    table=pd.DataFrame(data,index=h.index)
    table['truth']=h[target].shift(-horizon)
    table['label_available_at']=h.index+pd.Timedelta(hours=horizon)
    return table.dropna()

def block_skill_ci(frame,seed=42):
    """Resample calendar-day blocks, retaining within-day temporal dependence."""
    z=frame.copy();z['block']=pd.to_datetime(z.origin).dt.floor('1D')
    z['e_model']=abs(z.truth-z.prediction);z['e_base']=abs(z.truth-z.baseline)
    agg=z.groupby('block')[['e_model','e_base']].sum().to_numpy()
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(agg),size=(500,len(agg)))
    sums=agg[idx].sum(axis=1)
    valid=sums[:,1]>0
    skills=1-sums[valid,0]/sums[valid,1]
    return np.quantile(skills,[.025,.975]) if len(skills) else np.array([np.nan,np.nan])

def run_forecasting(out):
    records=[];preds=[];counts=[];parts=[];hourlies=[]
    for sensor in ['sensor01','sensor02','sensor03']:
        h=hourly_link(sensor);hourlies.append(h.assign(sensor=sensor).reset_index())
        cut1=h.index[int(.6*len(h))];cut2=h.index[int(.8*len(h))]
        for target in ['rssi','snr']:
            for horizon in [1,6]:
                table=forecast_table(h,target,horizon)
                tr=table[(table.index<cut1)&(table.label_available_at<cut1)]
                va=table[(table.index>=cut1)&(table.index<cut2)&(table.label_available_at<cut2)]
                te=table[table.index>=cut2]
                assert min(len(tr),len(va),len(te))>50
                columns=[c for c in table if '_lag' in c];uni=[c for c in columns if c.startswith(target+'_')]
                counts.append(dict(sensor=sensor,target=target,horizon=horizon,hours=len(h),
                    hours_with_reception=int(h.n_received.gt(0).sum()),complete_examples=len(table),
                    n_train=len(tr),n_validation=len(va),n_test=len(te),validation_start=str(cut1),test_start=str(cut2)))
                for label,frame in [('train',tr),('validation',va),('test',te)]:
                    parts.extend(dict(sensor=sensor,target=target,horizon=horizon,split=label,
                        origin=str(t),label_available_at=str(a),validation_start=str(cut1),test_start=str(cut2))
                        for t,a in zip(frame.index,frame.label_available_at))
                baseline=te[f'{target}_lag0'].to_numpy()
                outputs={'persistence':(baseline,None),'mean_24h':(te[uni].mean(axis=1).to_numpy(),None)}
                for name,cols in [('ridge_univariate',uni),('ridge_rssi_snr',columns)]:
                    candidates=[]
                    for alpha in [.1,1,10,100]:
                        m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(tr[cols],tr.truth)
                        candidates.append((mean_absolute_error(va.truth,m.predict(va[cols])),alpha,m))
                    _,alpha,m=min(candidates,key=lambda z:z[0])
                    outputs[name]=(m.predict(te[cols]),alpha)
                outputs['random_forest']=(forest().fit(tr[uni],tr.truth).predict(te[uni]),None)
                base_mae=mean_absolute_error(te.truth,baseline)
                for model,(p,alpha) in outputs.items():
                    frame=pd.DataFrame(dict(origin=te.index.astype(str),label_available_at=te.label_available_at.astype(str),
                        truth=te.truth.to_numpy(),prediction=p,baseline=baseline))
                    ci=block_skill_ci(frame) if model=='ridge_rssi_snr' else [np.nan,np.nan]
                    score=metrics(te.truth,p)
                    records.append(dict(sensor=sensor,target=target,horizon=horizon,model=model,n_test=len(te),alpha=alpha,
                        skill_mae=1-score['mae']/base_mae if base_mae else np.nan,
                        skill_ci_low=ci[0],skill_ci_high=ci[1],**score))
                    preds.append(frame.assign(sensor=sensor,target=target,horizon=horizon,model=model))
                print('Forecast',sensor,target,horizon,flush=True)
    pd.DataFrame(records).to_csv(out/'forecast_metrics.csv',index=False)
    pd.concat(preds).to_csv(out/'forecast_predictions.csv.gz',index=False)
    pd.DataFrame(counts).to_csv(out/'forecast_eligibility.csv',index=False)
    pd.DataFrame(parts).to_csv(out/'forecast_splits.csv.gz',index=False)
    pd.concat(hourlies).to_csv(out/'uva_hourly.csv',index=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=['all','localization','pathloss','forecasting'],default='all')
    parser.add_argument('--output',default='results/pilot_v1');args=parser.parse_args()
    out=ROOT/args.output
    if out.exists() and any(out.iterdir()): raise SystemExit('Choose a new output directory; previous results are preserved.')
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    selected={'localization':run_localization,'pathloss':run_pathloss,'forecasting':run_forecasting}
    for name,fn in selected.items():
        if args.study in ['all',name]: fn(out)
    sources=[Path(__file__),ROOT/'docs/PROTOCOLO_PILOTO.md',ROOT/'data/manifest.json']
    manifest=dict(study=args.study,elapsed_seconds=time.time()-start,python=platform.python_version(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow']},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
