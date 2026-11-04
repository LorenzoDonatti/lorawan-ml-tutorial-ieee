"""One inspectable path through the existing sensor01 forecasting experiment.

This replays the declared anchor protocol; it adds no benchmark condition.
The notebook displays the preparation/selection functions as executable cells.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from revision_models import fit, configurations

ROOT=Path(__file__).resolve().parents[1]


def load_records(root):
    raw=pd.read_parquet(root/'data/raw/uva/sensor01_gatewayA.parquet')
    receptions=raw.set_index('Timestamp').sort_index()
    hourly=receptions[['RSSI (dBm)','SNR (dB)']].rename(
        columns={'RSSI (dBm)':'rssi','SNR (dB)':'snr'}).resample('1h').mean()
    hourly['rssi_count']=receptions['RSSI (dBm)'].resample('1h').count()
    hourly['snr_count']=receptions['SNR (dB)'].resample('1h').count()
    # A value stamped 20:00 summarizes [19:00, 20:00), in UTC.
    hourly.index=hourly.index+pd.Timedelta(hours=1)
    return raw,hourly


def make_examples(hourly,window=24,horizon=6):
    if not isinstance(window,int) or not isinstance(horizon,int) or min(window,horizon)<1:
        raise ValueError('History and horizon must be positive integer hours.')
    frame=pd.DataFrame(index=hourly.index)
    for variable in ['rssi','snr']:
        for lag in range(window):
            frame[f'{variable}_lag{lag}']=hourly[variable].shift(lag)
    features=list(frame.columns)
    frame['truth']=hourly.snr.shift(-horizon)
    frame['label_available_at']=frame.index+pd.Timedelta(hours=horizon)
    c1,c2=[hourly.index[int(f*len(hourly))] for f in [.6,.8]]
    end=hourly.index[-1]+pd.Timedelta(hours=1)
    history_ready=frame[features].notna().all(axis=1)
    eligible=history_ready & frame.truth.notna()
    frame['split']='excluded'
    for name,mask in [('train',(frame.index<c1)&(frame.label_available_at<c1)),
                      ('validation',(frame.index>=c1)&(frame.index<c2)&(frame.label_available_at<c2)),
                      ('test',(frame.index>=c2)&(frame.label_available_at<end))]:
        frame.loc[mask & eligible,'split']=name
    return frame,features,dict(train_end=str(c1),validation_end=str(c2),test_end=str(end))


def mae(truth,prediction):
    return float(np.abs(np.asarray(truth)-np.asarray(prediction)).mean())


def select_on_validation(frame,features,window=24):
    # Test rows are not passed to fitting, stopping or candidate selection.
    tr=frame[frame.split.eq('train')];va=frame[frame.split.eq('validation')]
    models={};selected={};candidates=[]
    for family in ['ridge','mlp']:
        best=None
        for candidate,config in enumerate(configurations(family)):
            fitted,history,info=fit(tr[features].to_numpy(),tr[['truth']].to_numpy(),
                va[features].to_numpy(),va[['truth']].to_numpy(),family=family,
                config=config,seed=42,score=mae,window=window)
            value=mae(va[['truth']].to_numpy(),fitted.predict(va[features].to_numpy()))
            candidates.append(dict(model=family,candidate=candidate,config=json.dumps(config,sort_keys=True),validation_mae=value))
            if best is None or value<best[0]:best=(value,fitted,config,history)
        models[family]=best[1];selected[family]=best[2]
    return models,selected,pd.DataFrame(candidates)


def evaluate(frame,features,models,window=24):
    test=frame[frame.split.eq('test')]
    predictions=pd.DataFrame(index=test.index)
    predictions['truth']=test.truth
    predictions['persistence']=test.snr_lag0
    predictions['window_mean']=test[[f'snr_lag{k}' for k in range(window)]].mean(axis=1)
    for family,fitted in models.items():predictions[family]=fitted.predict(test[features].to_numpy()).ravel()
    rows=[]
    for model in ['persistence','window_mean','ridge','mlp']:
        error=predictions[model]-predictions.truth
        rows.append(dict(model=model,mae=float(error.abs().mean()),rmse=float(np.sqrt((error**2).mean()))))
    metrics=pd.DataFrame(rows)
    metrics['mae_minus_mean']=metrics.mae-float(metrics.loc[metrics.model.eq('window_mean'),'mae'].iloc[0])
    return predictions,metrics


def verify_protocol(hourly,frame,features,boundaries,window,horizon):
    """Check a declared forecast problem without requiring published scores.

    These conditions also apply to reader adaptations. The fixed-anchor check
    below has a different purpose: reproducing the original six-hour study.
    """
    assert frame.index.equals(hourly.index), 'Keep the complete hourly calendar.'
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert frame.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    expected=[f'{v}_lag{k}' for v in ['rssi','snr'] for k in range(window)]
    assert features==expected, 'Feature names and order must match the declared history.'
    assert set(frame.split.unique())<=set(['train','validation','test','excluded'])
    for variable in ['rssi','snr']:
        for lag in range(window):
            stamps=frame.index-pd.Timedelta(hours=lag)
            values=hourly[variable].reindex(stamps).to_numpy()
            np.testing.assert_allclose(frame[f'{variable}_lag{lag}'],values,equal_nan=True,
                err_msg=f'{variable}_lag{lag} must use observations available at the origin.')
    target_times=frame.index+pd.Timedelta(hours=horizon)
    assert pd.DatetimeIndex(frame.label_available_at).equals(target_times), 'Incorrect label-availability time.'
    np.testing.assert_allclose(frame.truth,hourly.snr.reindex(target_times).to_numpy(),equal_nan=True,
        err_msg='The target must refer to the declared future hour.')
    ready=frame[features].notna().all(axis=1)&frame.truth.notna()
    c1=pd.Timestamp(boundaries['train_end']);c2=pd.Timestamp(boundaries['validation_end'])
    end=pd.Timestamp(boundaries['test_end'])
    assert frame.index[0]<c1<c2<end
    expected_split=pd.Series('excluded',index=frame.index)
    for phase,start,stop in [('train',frame.index[0],c1),('validation',c1,c2),('test',c2,end)]:
        allowed=ready&(frame.index>=start)&(frame.index<stop)&(target_times<stop)
        expected_split.loc[allowed]=phase
    assert frame.split.equals(expected_split), 'Partition membership must respect eligibility and label arrival.'
    retained=frame[frame.split.ne('excluded')]
    assert np.isfinite(retained[features+['truth']].to_numpy()).all()
    counts={p:int(frame.split.eq(p).sum()) for p in ['train','validation','test']}
    assert min(counts.values())>0, 'Each phase needs evaluable examples.'
    return dict(status='passed',window_hours=window,horizon_hours=horizon,counts=counts,
                scope='Calendar, feature availability, target alignment and partitions; no fixed scores required')


def verify(root,frame,features,selected,predictions):
    """Verify the published 24-hour-history, six-hour-horizon anchor only."""
    reference=root/'results/full_repetition/controlled'
    a=np.load(reference/'inputs/sensor01_anchor.npz')
    retained=frame[~frame.split.eq('excluded')]
    np.testing.assert_allclose(retained[features],a['X'],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(retained.split,a['split'])
    np.testing.assert_array_equal(retained.index.astype(str),a['row_id'])
    chosen=json.loads((reference/'selection.json').read_text())
    for family,config in selected.items():assert config==chosen['sensor01_anchor__'+family]['config']
    old=pd.read_csv(reference/'predictions.csv.gz',low_memory=False)
    checks={}
    for model in ['persistence','window_mean','ridge','mlp']:
        q=old[old.case.eq('sensor01_anchor')&old.model.eq(model)&old.phase.eq('test')]
        if model in ['ridge','mlp']:q=q[q.seed.eq(42)]
        q=q.copy();q.index=pd.to_datetime(q.row_id,utc=True);q=q.loc[predictions.index]
        np.testing.assert_allclose(predictions[model],q.prediction_0,rtol=1e-7,atol=1e-7)
        checks[model]=float(np.max(np.abs(predictions[model]-q.prediction_0)))
    return dict(status='passed',same_inputs_partitions_and_selected_settings=True,
                predictions=len(predictions),max_prediction_difference=checks,
                interpretation='Replay of existing seed-42 anchor; not new independent evidence')


def table(path,caption,label,columns,rows,fmt):
    text='\\begin{table*}[t]\\centering\\small\n'
    text+=f'\\caption{{{caption}}}\\label{{tab:{label}}}\n'
    text+=f'\\begin{{tabular}}{{{fmt}}}\\toprule\n'+' & '.join(columns)+' \\\\\\midrule\n'
    text+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)
    path.write_text(text+'\n\\bottomrule\\end{tabular}\\end{table*}\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['reproduce','adapt'],default='reproduce')
    p.add_argument('--window',type=int,default=24)
    p.add_argument('--horizon',type=int,default=6)
    p.add_argument('--write-paper',action='store_true',help='Regenerate paper fragments in reproduce mode only.')
    a=p.parse_args()
    if a.mode=='reproduce' and (a.window,a.horizon)!=(24,6):
        p.error('A changed history or horizon requires --mode adapt.')
    if a.write_paper and a.mode!='reproduce':p.error('Adaptations cannot overwrite manuscript results.')
    a.output.mkdir(parents=True,exist_ok=False)
    raw,hourly=load_records(ROOT);frame,features,boundaries=make_examples(hourly,a.window,a.horizon)
    protocol=verify_protocol(hourly,frame,features,boundaries,a.window,a.horizon)
    models,selected,candidates=select_on_validation(frame,features,window=a.window)
    predictions,metrics=evaluate(frame,features,models,window=a.window)
    check=verify(ROOT,frame,features,selected,predictions) if a.mode=='reproduce' else dict(
        status='passed',interpretation='Teaching adaptation; not a published comparison or independent benchmark')
    check.update(protocol=protocol,mode=a.mode,boundaries=boundaries,selected=selected,
                 source_sha256=hashlib.sha256((ROOT/'data/raw/uva/sensor01_gatewayA.parquet').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    candidates.to_csv(a.output/'candidates.csv',index=False);metrics.to_csv(a.output/'metrics.csv',index=False)
    predictions.to_csv(a.output/'predictions.csv.gz',index_label='origin')
    frame[['label_available_at','split']].to_csv(a.output/'partitions.csv.gz',index_label='origin')
    (a.output/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    if not a.write_paper:
        print(metrics.to_string(index=False));print(json.dumps(check,indent=2));return
    t=predictions.index[0]  # first eligible test origin; no error-based selection
    sample=raw[(raw.Timestamp>=t-pd.Timedelta(hours=1))&(raw.Timestamp<t)].sort_values('Timestamp')
    sample.to_csv(a.output/'raw_hour.csv',index=False)
    window=hourly.loc[t-pd.Timedelta(hours=23):t].copy();window.to_csv(a.output/'history.csv',index_label='closed_utc')
    g=ROOT/'paper/generated'
    table(g/'running_raw.tex',f'Running example: first three receptions in the input hour closing at {t:%Y-%m-%d %H:%M} UTC. These are actual source rows, not the full hour. Means use all nonmissing receptions of each signal.',
          'runningraw',['Reception time (UTC)','RSSI (dBm)','SNR (dB)','Frame counter'],
          [[r.Timestamp.strftime('%H:%M:%S'),f'{r["RSSI (dBm)"]:.1f}',('Missing' if pd.isna(r["SNR (dB)"]) else f'{r["SNR (dB)"]:.1f}'),int(r['Counter (-)'])] for _,r in sample.head(3).iterrows()], 'lrrr')
    rows=[]
    for lag in [23,1,0]:
        stamp=t-pd.Timedelta(hours=lag);r=hourly.loc[stamp]
        rows.append([f'$t-{lag}$' if lag else '$t$',stamp.strftime('%m-%d %H:%M'),f'{r.rssi:.3f}',f'{r.snr:.3f}','Yes'])
    stamp=t+pd.Timedelta(hours=6)
    rows.append(['$t+6$',stamp.strftime('%m-%d %H:%M'),'Withheld',f'{hourly.loc[stamp,"snr"]:.3f}','No: target only'])
    table(g/'running_window.tex','Selected rows of the actual 24-hour history and its future target, in UTC in 2025. The omitted 21 input hours are retained in the calculation and notebook; the future SNR is shown only to audit the later outcome.',
          'runningwindow',['Role in time','Hour closes','RSSI (dBm)','SNR (dB)','Available at issue?'],rows,'llrrl')
    labels={'persistence':'Persistence','window_mean':'24-hour mean','ridge':'Ridge','mlp':'MLP (seed 42)'}
    table(g/'running_results.tex','Running-example test errors on the same 1,937 eligible sensor01 origins. Positive differences in the last column favor the historical mean. The MLP row is the declared teaching initialization; Section XI reports all declared initializations.',
          'runningresults',['Procedure','MAE (dB)','RMSE (dB)','$\\Delta$MAE vs. mean (dB)'],
          [[labels[r.model],f'{r.mae:.3f}',f'{r.rmse:.3f}',f'{r.mae_minus_mean:+.3f}'] for r in metrics.itertuples()], 'lrrr')
    r=predictions.iloc[0];m=hourly.loc[t]
    (g/'running_values.tex').write_text(
        f'At the first eligible test origin, {t:%Y-%m-%d %H:%M} UTC, the latest input hour contains {int(m.rssi_count)} RSSI and {int(m.snr_count)} SNR values. '
        f'Its SNR mean is {m.snr:.3f} dB. Persistence predicts {r.persistence:.3f} dB and the 24-hour mean predicts {r.window_mean:.3f} dB; '
        f'the subsequently observed target is {r.truth:.3f} dB. Their absolute errors for this single example are {abs(r.persistence-r.truth):.3f} and {abs(r.window_mean-r.truth):.3f} dB, respectively. '
        'One example teaches the calculation; the held-out cohort determines the reported comparison.\n')
    (g/'running_boundaries.tex').write_text(
        f'The training/validation boundary is {pd.Timestamp(boundaries["train_end"]):%Y-%m-%d %H:%M} UTC; the validation/test boundary is {pd.Timestamp(boundaries["validation_end"]):%Y-%m-%d %H:%M} UTC. '
        'After availability filtering, the partitions contain 6,917 training, 2,211 validation and 1,937 test origins. '
        'For example, an origin exactly six hours before the first boundary is excluded from training: its label arrives at that boundary, rather than before it.\n')
    check.update(boundaries=boundaries,origin=str(t),selected=selected,source_sha256=hashlib.sha256((ROOT/'data/raw/uva/sensor01_gatewayA.parquet').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(metrics.to_string(index=False));print(json.dumps(check,indent=2))


if __name__=='__main__':main()
