"""Bounded GB128 C256 checkpoint-only comparison, executed on eight GPUs.

Same script embedded in job 2100507732791533568. Each variant starts with
fresh weights/optimizer twice; this is not formal training continuation.
"""
import json,os,pathlib,signal,subprocess,sys,time
out=pathlib.Path('/trisol/output/checkpoint-validation');out.mkdir(parents=True,exist_ok=True)
summary=[]
for variant in ['serial-m2-cp','serial-m2-nocp']:
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','-m','ouro_depth.latent.profile_stage2','--mode','update','--variant',variant,'--chunk','256','--output',str(out),'--repeats','2','--legacy-groups']
 print('FULL_UPDATE_START '+json.dumps(cmd),flush=True)
 with (out/(variant+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=900)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 rows=[]
 for path in sorted(out.glob(variant+'-c256-rank*.jsonl')):
  for line in path.read_text().splitlines():
   row=json.loads(line)
   if row['event'] in ['ready','result']:rows.append(row)
 entry=dict(variant=variant,exit_code=code,rows=rows)
 if code:entry['failure_tail']=(out/(variant+'.log')).read_text()[-6000:]
 summary.append(entry)
 print('FULL_UPDATE_RESULT '+json.dumps(entry),flush=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
print('FULL_UPDATE_COMPLETE',flush=True)
if any(x['exit_code'] for x in summary):sys.exit(1)
