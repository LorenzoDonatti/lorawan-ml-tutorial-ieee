from pathlib import Path
import argparse,json,time
import nbformat
from nbclient import NotebookClient
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--inprocess',action='store_true')
parser.add_argument('--pattern',default='*.ipynb');args=parser.parse_args()
records=[]
log=ROOT/'results/notebook_execution.json'
prior=json.loads(log.read_text()) if log.exists() else []
updated={r['notebook']:r for r in prior}
for p in sorted((ROOT/'notebooks').glob(args.pattern)):
    nb=nbformat.read(p,as_version=4);start=time.time()
    if args.inprocess:
        # The managed workspace disallows kernel sockets. Execute all cells
        # in a fresh IPython namespace, retaining rich outputs, without sockets.
        InteractiveShell.clear_instance();shell=InteractiveShell.instance()
        matplotlib.use('module://matplotlib_inline.backend_inline')
        count=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1
            with capture_output() as cap:
                result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec:raise result.error_before_exec
            if result.error_in_exec:raise result.error_in_exec
            cell.execution_count=count;cell.outputs=[]
            if cap.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
            if cap.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
            for o in cap.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata))
    else:
        NotebookClient(nb,timeout=300,kernel_name='lorawan-tutorial',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,p)
    records.append(dict(notebook=p.name,seconds=time.time()-start,status='executed',engine='IPython in-process' if args.inprocess else 'Jupyter kernel',cells=len(nb.cells)))
    print(records[-1],flush=True)
    updated[p.name]=records[-1]
    log.write_text(json.dumps(sorted(updated.values(),key=lambda r:r['notebook']),indent=2)+'\n')
