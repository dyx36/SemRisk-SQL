"""Conservative component evidence. Features are not correctness labels."""
import copy
from collections import Counter
import constraints as c
from local_evidence import normalize_expression
GROUPS=("output_columns","output_operators","count_target","count_distinct","grouping_grain","ratio_structure",
 "ratio_numerator_filter","ratio_denominator_leakage","filter_column","filter_operator","filter_value","order_key_direction","row_limit")
STATES=("supported","contradicted","unknown")
STRUCTURES=("top1_single_order_key","output_distinct","has_join","count_distinct_over_join","ratio_integer_division_possible")
FEATURE_NAMES=tuple(f"{g}_{s}" for g in GROUPS for s in STATES)+STRUCTURES
AGG={"count","sum","avg","min","max"};CMP={"eq","neq","gt","gte","lt","lte"}
def repair_expression(x):
 if not isinstance(x,dict):return x
 x=copy.deepcopy(x);op=x.get("op");args=x.get("args")
 if isinstance(args,list):
  # Unambiguous redundant infix operator, e.g. eq(column,"eq",literal).
  if op in CMP and len(args)==3 and args[1]==op:args=[args[0],args[2]]
  # One exact alternate notation: count(filter(column,op,value),counted_expression).
  if op in AGG and len(args)==2 and isinstance(args[0],dict) and args[0].get("op")=="filter":
   filt=args[0].get("args")
   if isinstance(filt,list) and len(filt)==3 and isinstance(filt[1],str) and filt[1] in CMP:
    x["args"]=[repair_expression(args[1])];x["check_population"]=True
    x["population_filter"]=repair_expression({"op":filt[1],"args":[filt[0],filt[2]]})
    return normalize_expression(x)
  x["args"]=[repair_expression(a) for a in args]
 if isinstance(x.get("population_filter"),dict):x["population_filter"]=repair_expression(x["population_filter"])
 return normalize_expression(x)
def requirements(raw,q,schema,visible):
 raw=copy.deepcopy(raw)
 for entries in raw.values():
  if not isinstance(entries,list):continue
  for r in entries:
   if not isinstance(r,dict):continue
   expr=repair_expression(r.get("expression"))
   if r.get("kind")=="count" and isinstance(expr,dict) and expr.get("op")=="count":
    for key in ("distinct","grain","check_population","population_filter"):
     if key in r and key not in expr:expr[key]=repair_expression(r[key])
   r["expression"]=expr
 return c.bind_requirements(raw,q,schema,visible)
def atoms(x):
 if not isinstance(x,dict):return []
 if x.get("op")=="and":return [a for child in x.get("args",[]) for a in atoms(child)]
 return [x]
def columns(x):return {c.column_key(n) for n in c.walk(x) if n.get("op")=="column"}
def equivalent(a,b):
 if a is None or b is None:return False
 aa=c.shape(a);bb=c.shape(b);return aa is not None and bb is not None and aa==bb
def predicate_parts(x):
 if not isinstance(x,dict) or x.get("op") not in CMP:return None
 a=x.get("args",[])
 if len(a)!=2:return None
 if not all(isinstance(v,dict) for v in a):return None
 if a[0].get("op")=="column" and a[1].get("op")=="literal":return c.column_key(a[0]),x["op"],c.lit(a[1])
 if a[1].get("op")=="column" and a[0].get("op")=="literal":return c.column_key(a[1]),{"gt":"lt","gte":"lte","lt":"gt","lte":"gte"}.get(x["op"],x["op"]),c.lit(a[0])
 return None
def meaningful_nodes(expr):
 # Population metadata is not part of the projected arithmetic expression.
 if not isinstance(expr,dict):return []
 return [expr]+[n for a in expr.get("args",[]) for n in meaningful_nodes(a)]
def observe_document(doc,req):
 counts=Counter();ledger=[];current={}
 def add(group,state,reason):counts[group,state]+=1;ledger.append(dict(current,component=group,state=state,reason=reason))
 def truth(group,condition,reason):add(group,"supported" if condition else "contradicted",reason)
 root=next((s for s in doc.get("scopes",[]) if s["id"]==doc.get("root_scope")),{})
 ambiguous=(doc.get("status")!="parsed" or any(s.get("derived") for s in root.get("sources",[]))
  or any(n>1 for n in Counter(s.get("table") for s in root.get("sources",[])).values()))
 outputs=doc.get("outputs",[]);actual_nodes=[n for o in outputs for n in meaningful_nodes(o)];actual_agg=[n for n in actual_nodes if n.get("op") in AGG]
 actual_cols=set().union(*(columns(o) for o in outputs)) if outputs else set();actual_ops={n.get("op") for n in actual_nodes}
 for family,entries in req.items():
  for requirement_index,r in enumerate(entries or []):
   current={"family":family,"requirement":requirement_index}
   expr=r.get("expression");kind=r.get("kind")
   relevant={"output":["output_columns","output_operators"],"count":["count_target","count_distinct","grouping_grain"],
    "ratio":["ratio_structure","ratio_numerator_filter","ratio_denominator_leakage"],"filter":["filter_column","filter_operator","filter_value"],
    "order":["order_key_direction"],"limit":["row_limit"]}.get(kind,[])
   if ambiguous or (expr is None and kind!="limit"):
    for group in relevant:add(group,"unknown","unresolved_binding_or_scope")
    continue
   if kind=="output":
    expected_cols=columns(expr)
    if expected_cols:truth("output_columns",expected_cols<=actual_cols,"expected_projection_columns_present")
    ops={n.get("op") for n in meaningful_nodes(expr)}-{"column","literal","star"}
    if ops:truth("output_operators",ops<=actual_ops,"requested_operator_atoms_present_not_full_equivalence")
   if kind=="count":
    expected=[n for n in meaningful_nodes(expr) if n.get("op")=="count"]
    for e in expected:
     candidates=[a for a in actual_agg if a.get("op")=="count"]
     matches=[a for a in candidates if equivalent(e.get("args",[None])[0],a.get("args",[None])[0])]
     if matches:add("count_target","supported","count_argument_matches")
     elif not candidates:add("count_target","contradicted","requested_count_absent")
     else:add("count_target","unknown","count_star_key_null_equivalence_not_proven")
     if e.get("distinct") is not None and matches:truth("count_distinct",any(bool(a.get("distinct"))==e["distinct"] for a in matches),"explicit_distinct_matches_on_same_target")
     if e.get("grain") is not None:
      g=[c.shape(v) for v in e["grain"]];a=[c.shape(v) for v in doc.get("groups",[])]
      if all(v is not None for v in g+a):truth("grouping_grain",set(g)==set(a),"explicit_grouping_keys_match")
   if kind=="filter":
    candidates=atoms(doc.get("filter"));parts=[predicate_parts(a) for a in candidates];parts=[a for a in parts if a is not None]
    unsafe=any(n.get("op") in ("or","not","unknown","exists","subquery","in","between","like","is","is_not") for n in c.walk(doc.get("filter")))
    for expected in atoms(expr):
     e=predicate_parts(expected)
     if e is None or unsafe:
      for group in ("filter_column","filter_operator","filter_value"):add(group,"unknown","boolean_or_predicate_form_unresolved")
      continue
     samecol=[v for v in parts if v[0]==e[0]];sameop=[v for v in samecol if v[1]==e[1]]
     truth("filter_column",bool(samecol),"explicit_filter_column_present")
     if samecol:truth("filter_operator",bool(sameop),"filter_operator_matches_on_same_column")
     if sameop:truth("filter_value",any(v[2]==e[2] for v in sameop),"filter_literal_matches_on_same_column_operator")
   if kind=="ratio":
    ex=[n for n in meaningful_nodes(expr) if n.get("op")=="divide"];ac=[n for n in actual_nodes if n.get("op")=="divide"]
    if ex:truth("ratio_structure",bool(ac),"explicit_formula_requires_division")
    for e in ex:
     if len(e.get("args",[]))!=2:continue
     num,den=e["args"];en=[n for n in meaningful_nodes(num) if n.get("op") in AGG];ed=[n for n in meaningful_nodes(den) if n.get("op") in AGG]
     numerator_filters=[v for n in en for v in atoms(n.get("population_filter"))]
     denominator_filters=[v for n in ed for v in atoms(n.get("population_filter"))]
     exclusive=[v for v in numerator_filters if not any(equivalent(v,t) for t in denominator_filters)]
     if len(ac)!=1 or len(ac[0].get("args",[]))!=2:
      if exclusive:add("ratio_denominator_leakage","unknown","ratio_population_not_resolved")
      continue
     an,ad=ac[0]["args"];an_filters=[v for n in meaningful_nodes(an) if n.get("op") in AGG for v in atoms(n.get("population",{}).get("filter"))]
     ad_nodes=[n for n in meaningful_nodes(ad) if n.get("op") in AGG];ad_filters=[v for n in ad_nodes for v in atoms(n.get("population",{}).get("filter"))]
     for efilter in numerator_filters:truth("ratio_numerator_filter",any(equivalent(efilter,v) for v in an_filters),"explicit_numerator_predicate_present")
     for efilter in exclusive:
      if ad_nodes:truth("ratio_denominator_leakage",not any(equivalent(efilter,v) for v in ad_filters),"numerator_only_predicate_not_applied_to_denominator")
   if kind in ("order","limit"):
    state,reason=c.compare_requirement(r,doc);add("order_key_direction" if kind=="order" else "row_limit",state,reason)
 flags=[float(c.lit(doc.get("limit"))==("number","1") and len(doc.get("orders",[]))==1),float(bool(doc.get("distinct"))),
  float(bool(root.get("joins"))),float(bool(root.get("joins")) and any(n.get("op")=="count" and n.get("distinct") for n in actual_agg)),
  float(any(n.get("op")=="divide" and n.get("division_semantics")=="integer_possible" for n in actual_nodes))]
 return [float(counts[g,s]) for g in GROUPS for s in STATES]+flags,{"status":doc.get("status"),"component_observations":ledger,
  "warning":"component agreement is partial; structure flags are ambiguity/descriptive signals, not correctness"}
