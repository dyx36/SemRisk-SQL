"""Frozen 105-dimension feature contract and label-free official-pool extraction."""
import signal,time
from pathlib import Path
import numpy as np
import constraints as c
import local_evidence as old
import contrast_elements as new
from common import *
FEATURE_NAMES=old.FEATURE_NAMES+new.FEATURE_NAMES+tuple(f"peer_centered_{g}_{s}" for g in new.GROUPS for s in ("supported","contradicted"))
SOURCE=ROOT/'.codex_runs/learnrisk_local_evidence_v1/run'
def assemble(old_features,components):
 a=np.asarray(components,dtype=np.float32);assert a.shape==(30,44)
 v=a[:,:39].reshape(30,13,3);s=v[:,:,0];t=v[:,:,1]
 centered=np.stack((s-s.mean(0,keepdims=True),t-t.mean(0,keepdims=True)),axis=-1).reshape(30,26)
 out=np.concatenate((old_features,a,centered),axis=1);assert out.shape==(30,105)
 return out
def ensure_dev():
 if (RUN/'dev_features_ready.json').exists():return
 verify_source();qs=list(jsonlines(RUN/'data/questions_dev.jsonl'));rows=list(jsonlines(RUN/'data/candidates_dev.jsonl'))
 assert len(qs)==1534 and len(rows)==46020 and len(FEATURE_NAMES)==105
 prior_features=np.load(SOURCE/'dev_features.npy');old_ready=read(SOURCE/'dev_features_ready.json')
 assert sha(SOURCE/'dev_features.npy')==old_ready['matrix_sha256'];schemas={};matrix=[]
 for k,q in enumerate(qs):
  out=RUN/'component_observations'/('dev_'+str(q['question_id'])+'.json')
  if not out.exists():
   if q['db_id'] not in schemas:schemas[q['db_id']]=read(RUN/'data/schemas'/('dev_'+q['db_id']+'.json'))
   schema=schemas[q['db_id']];source=SOURCE/'contracts'/('dev_'+str(q['question_id'])+'.json');contract=read(source)
   assert contract['question_sha256']==digest(q) and contract['schema_sha256']==digest(schema)
   raw=c.parse_partial_object(contract['raw_response']['choices'][0]['message'].get('content') or '')
   req,audit=new.requirements(raw,q,schema,set(contract['visible_column_ids']));cache={};features=[]
   for i,row in enumerate(rows[k*30:(k+1)*30]):
    assert row['question_id']==q['question_id'] and row['candidate_index']==i
    key=row['sql_sha256']
    if key not in cache:
     def alarm(*args):raise TimeoutError('parser_budget_10s')
     previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(10)
     try:cache[key]=new.observe_document(c.analyze_candidate(row['sql'],{'tables':schema['tables'],'foreign_keys':schema['foreign_keys']}),req)[0]
     except (TimeoutError,RecursionError):cache[key]=[float(j<39 and j%3==2) for j in range(44)]
     finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    features.append(cache[key])
   write(out,dict(question_id=q['question_id'],components=features,source_contract_sha256=sha(source),labels_used=False))
  values=read(out)['components'];matrix.extend(assemble(prior_features[k*30:(k+1)*30],values))
  if (k+1)%100==0 or k==0:event('CONTRAST_DEV_FEATURE_PROGRESS',questions=k+1,total=1534)
 a=np.asarray(matrix,dtype=np.float32);assert a.shape==(46020,105) and np.isfinite(a).all()
 np.save(RUN/'dev_features.npy',a)
 # Existing compiler-invalid flags and index are copied by SHA from the original pool.
 write(RUN/'dev_features_ready.json',dict(questions=1534,candidates=46020,features=FEATURE_NAMES,matrix_sha256=sha(RUN/'dev_features.npy'),index_sha256=sha(RUN/'dev_index.jsonl'),labels_used=False))
 event('CONTRAST_DEV_FEATURES_COMPLETE')
