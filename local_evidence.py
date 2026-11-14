"""Cached source-bound requirements, lossless literal normalization and typed evidence."""
import copy,json
from collections import Counter
from pathlib import Path
import constraints as c
KINDS=("count","ratio","output","order","limit","filter","relation")
STATES=("supported","contradicted","unknown")
REASONS=("requirement_binding_unresolved","output_lineage_unresolved","repeated_table_role_unresolved",
 "answer_scope_or_boolean_entailment_unresolved","relationship_path_unresolved","explicit_order_missing",
 "order_key_or_direction_differs","explicit_limit_missing","limit_differs","explicit_answer_filter_missing_or_differs")
FEATURE_NAMES=tuple(f"{k}_{s}" for k in KINDS for s in STATES)+tuple("reason_"+r for r in REASONS)+tuple(f"{f}_contract_missing" for f in c.FAMILIES)
def normalize_expression(x):
 if not isinstance(x,dict):return x
 out=copy.deepcopy(x)
 if isinstance(out.get("args"),list):
  args=[]
  for a in out["args"]:
   if type(a) in (str,int,float,bool):
    # Bare schema IDs are ambiguous; never reinterpret as literal column names.
    if isinstance(a,str) and (a=="*" or a.startswith("c") and a[1:].isdigit()):args.append(a)
    else:args.append({"op":"literal","value":a})
   else:args.append(normalize_expression(a))
  out["args"]=args
 if isinstance(out.get("population_filter"),dict):out["population_filter"]=normalize_expression(out["population_filter"])
 return out
def cached_contract(q,schema,run,digest,write):
 original=Path("/home/15t2/lsh/FINER/.codex_runs/learnrisk_prior_rules_full7700_20261023_v3/run/contracts")/(q["domain"]+"_"+str(q["question_id"])+".json")
 source=json.loads(original.read_text());assert source["question_sha256"]==digest(q) and source["schema_sha256"]==digest(schema)
 raw=c.parse_partial_object(source["raw_response"]["choices"][0]["message"].get("content") or "")
 raw=copy.deepcopy(raw)
 for entries in raw.values():
  if isinstance(entries,list):
   for r in entries:
    if isinstance(r,dict):r["expression"]=normalize_expression(r.get("expression"))
 requirements,audit=c.bind_requirements(raw,q,schema,set(source["visible_column_ids"]))
 result=dict(source,requirements=requirements,binding_audit=audit,cache_source=str(original),
             interpretation_revision="bare non-schema scalar expression args become explicit literals; no new model request")
 write(run/"contracts"/original.name,result);return result
def observe(sql,schema,requirements):
 doc=c.analyze_candidate(sql,{"tables":schema["tables"],"foreign_keys":schema["foreign_keys"]})
 counts=Counter();reasons=Counter();ledger=[];missing=[]
 for fam in c.FAMILIES:
  entries=requirements.get(fam);missing.append(float(entries is None))
  for i,r in enumerate(entries or []):
   kind=r.get("kind")
   if kind not in KINDS:continue
   state,reason=c.compare_requirement(r,doc)
   counts[kind,state]+=1;reasons[reason]+=1
   ledger.append({"family":fam,"requirement":i,"kind":kind,"state":state,"reason":reason})
 return [float(counts[k,s]) for k in KINDS for s in STATES]+[float(reasons[r]) for r in REASONS]+missing,{"status":doc["status"],"observations":ledger,"flags":doc.get("flags",[]),"capabilities":"same conservative scope-aware comparisons; typed requirement evidence; no semantic truth claim"}
