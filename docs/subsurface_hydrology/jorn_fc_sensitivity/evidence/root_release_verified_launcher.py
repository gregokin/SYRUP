from pathlib import Path
import os,sys,subprocess,json,time,datetime
f=Path(sys.argv[1]).resolve();cfg=json.loads(f.read_text());o=Path(cfg['output']);o.mkdir(exist_ok=False);env=os.environ.copy();env.update(cfg['env']);env['PYTHONPATH']=os.pathsep.join(cfg['prefix'])
if cfg.get('gpu_uuid'):
 r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);rows=[[x.strip() for x in line.split(',')] for line in r.stdout.strip().splitlines()];row=next(x for x in rows if x[0]=='0');(o/'gpu_guard.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'expected_uuid':cfg['gpu_uuid']},indent=2)+chr(10));assert row==['0',cfg['gpu_uuid'],'0','0'],row
cmd=[cfg['python'],'-B',str(f.parent/'root_release_verified_bootstrap.py'),str(f)];t=time.monotonic();record={'config':str(f),'config_sha256':__import__('hashlib').sha256(f.read_bytes()).hexdigest(),'expected_sources':cfg['sources'],'command':cmd,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'}
with (o/'out').open('w') as stdout,(o/'err').open('w') as stderr:
 q=subprocess.Popen(cmd,cwd=cfg['cwd'],env=env,stdout=stdout,stderr=stderr);st=Path('/proc')/str(q.pid)/'stat';parts=st.read_text().rsplit(')',1)[1].split();record.update(pid=q.pid,start_ticks=parts[19],ppid=int(parts[1]));(o/'result.json').write_text(json.dumps(record,indent=2)+chr(10));print('START',q.pid,flush=True);rc=q.wait()
record.update(status='closed',returncode=rc,seconds=time.monotonic()-t,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(o/'result.json').write_text(json.dumps(record,indent=2)+chr(10));print('DONE',rc,flush=True);raise SystemExit(rc)
