"""Four local, source-bound semantic observation families; no correctness labels."""
from __future__ import annotations
import json,re,sqlite3
from collections import Counter
from decimal import Decimal,InvalidOperation
from fractions import Fraction
from pathlib import Path
from risk_workflow.risk_v31.candidate_parser import analyze_candidate

FAMILIES=("count","ratio","selection","predicate")
FEATURE_NAMES=tuple(f"{f}_{s}_count" for f in FAMILIES for s in ("supported","contradicted","unknown"))
SYSTEM="""Extract ONLY explicit local obligations from QUESTION/EVIDENCE and visible SCHEMA.
Never generate SQL or judge candidates. Never invent a join, count key, DISTINCT, or scope.
Return one JSON object with four keys in this order: count, ratio, selection, predicate.
Each value is an array of at most 4 requirements, each with source ("question" or "evidence"),
quote (a SHORT EXACT substring from that source), kind, and the fields below.
Use schema column IDs c0,c1,...; unresolved bindings are null. [] means no explicit obligation.
All bindings remain model proposals. Do not infer certainty from a plausible schema match.

Expression syntax: {"op":"column","id":"c3"}; {"op":"literal","value":1};
{"op":"count|sum|avg|min|max|add|subtract|multiply|divide","args":[expressions]}.
For count/sum/avg/min/max additionally use distinct (true/false/null), grain (column expression array or null),
check_population (boolean), population_filter (boolean expression or null).
Use check_population=true only where the text determines the aggregate population; null filter then means
an explicitly unrestricted population, NOT an unresolved filter. Otherwise false.
Boolean expressions use eq|neq|gt|gte|lt|lte|and|or|not and args.
COUNT(*) uses {"op":"star"}. Never replace an entity count by arbitrary COUNT(*).
For a ratio describe the WHOLE requested formula, including percentage scale, numerator and denominator
population filters. Keep a numerator-only filter out of the denominator. Unknown formula parts are null.

count: kind="count", expression (requested count), optional quote for grouping in same requirement.
ratio: kind="ratio", expression (whole formula).
selection: kind="output", expression; OR kind="order", expression and direction ("asc"/"desc");
OR kind="limit", value (integer), only when explicitly requested.
predicate: kind="filter", expression (Boolean), ONLY a filter restricting ANSWER records;
OR kind="relation", expression (column comparison), only explicit relationship key binding.
Preserve AND/OR/NOT. Do not promote a local measure/CASE predicate to an answer filter.
Repeated-table entity roles that cannot be bound are null, never guessed.
Do not fill four arrays with generic requirements; only requirements justified by exact source quotes.
If a requested detail cannot be bound, include its source/quote with expression:null.
Return compact JSON only."""

def schema_for_db(path):
    con=sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)
    try:
        tables={}; fk=[]; ids={}; serial=0
        for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            columns=[]
            escaped=table.replace('"','""')
            for row in con.execute('PRAGMA table_info("'+escaped+'")'):
                cid="c"+str(serial);serial+=1
                col={"id":cid,"name":row[1],"type":row[2],"primary_key":bool(row[5]),"not_null":bool(row[3])}
                ids[cid]={"table":table.lower(),"column":row[1].lower(),"primary_key":bool(row[5]),"not_null":bool(row[3])}
                columns.append(col)
            tables[table]=columns
            for row in con.execute('PRAGMA foreign_key_list("'+escaped+'")'):
                if row[4] is not None:fk.append([table+"."+row[3],row[2]+"."+row[4]])
        return {"tables":tables,"foreign_keys":fk,"column_ids":ids}
    finally:con.close()

def prompt_payload(question,schema,char_budget=16000):
    terms=set(re.findall(r"[a-z0-9]+",question["question"].lower()+" "+question["evidence"].lower()))
    ranked=sorted(schema["tables"],key=lambda t:(-sum(len(set(re.findall(r"[a-z0-9]+",t+" "+c["name"].lower()))&terms) for c in schema["tables"][t]),t))
    visible={}; used=0
    for table in ranked:
        cols=schema["tables"][table]
        line=json.dumps([table,[[c["id"],c["name"],c["type"],int(c["primary_key"])] for c in cols]],ensure_ascii=False)
        if used+len(line)>char_budget:continue
        visible[table]=cols;used+=len(line)
    compact=[[t,[[c["id"],c["name"],c["type"],int(c["primary_key"])] for c in cols]] for t,cols in visible.items()]
    visible_ids={c["id"] for cols in visible.values() for c in cols}
    return {"question":question["question"],"evidence":question["evidence"],"schema":compact,
            "schema_truncated":len(visible)!=len(schema["tables"])},visible_ids

def parse_partial_object(text):
    # Decode only complete top-level values. Never repair or invent truncated JSON.
    decoder=json.JSONDecoder(); pos=0;result={}
    text=text.strip()
    if not text.startswith("{"):return result
    pos=1
    while pos<len(text):
        try:
            while pos<len(text) and text[pos].isspace():pos+=1
            if text[pos]=="}":break
            key,end=decoder.raw_decode(text,pos);pos=end
            while text[pos].isspace():pos+=1
            if not isinstance(key,str) or key in result or text[pos]!=":":break
            pos+=1
            while text[pos].isspace():pos+=1
            value,pos=decoder.raw_decode(text,pos);result[key]=value
            while pos<len(text) and text[pos].isspace():pos+=1
            if pos==len(text) or text[pos]=="}":break
            if text[pos]!=",":break
            pos+=1
        except (ValueError,IndexError):break
    return result

def bind_expression(obj,ids,visible,depth=0):
    if not isinstance(obj,dict) or depth>12:return None
    op=obj.get("op")
    if op=="column":
        cid=obj.get("id")
        return {"op":"column",**ids[cid]} if cid in visible and cid in ids else None
    if op=="star":return {"op":"star"}
    if op=="literal":
        v=obj.get("value")
        if v is None or type(v) in (str,int,float,bool):return {"op":"literal","value":v}
        return None
    if op not in {"count","sum","avg","min","max","add","subtract","multiply","divide","eq","neq","gt","gte","lt","lte","and","or","not"}:return None
    args=obj.get("args")
    if not isinstance(args,list) or not 1<=len(args)<=8:return None
    if op in {"add","subtract","multiply","divide","eq","neq","gt","gte","lt","lte"} and len(args)!=2:return None
    if op in {"count","sum","avg","min","max","not"} and len(args)!=1:return None
    args=[bind_expression(a,ids,visible,depth+1) for a in args]
    if any(a is None for a in args):return None
    out={"op":op,"args":args}
    if op in {"count","sum","avg","min","max"}:
        distinct=obj.get("distinct")
        if distinct is not None and type(distinct)is not bool:return None
        out["distinct"]=distinct
        grain=obj.get("grain")
        if grain is not None:
            if not isinstance(grain,list) or len(grain)>6:return None
            grain=[bind_expression(x,ids,visible,depth+1) for x in grain]
            if any(x is None or x["op"]!="column" for x in grain):return None
        out["grain"]=grain
        cp=obj.get("check_population",False)
        if type(cp)is not bool:return None
        out["check_population"]=cp
        pf=obj.get("population_filter")
        out["population_filter"]=bind_expression(pf,ids,visible,depth+1) if pf is not None else None
        if pf is not None and out["population_filter"] is None:return None
    return out

def bind_requirements(raw,question,schema,visible):
    result={}; audit=[]
    allowed={"count":{"count"},"ratio":{"ratio"},"selection":{"output","order","limit"},"predicate":{"filter","relation"}}
    for fam in FAMILIES:
        entries=raw.get(fam)
        if not isinstance(entries,list) or len(entries)>4:
            result[fam]=None;continue
        validated=[]
        for r in entries:
            if not isinstance(r,dict):audit.append({"family":fam,"reason":"requirement_not_object"});continue
            field=r.get("source");quote=r.get("quote");kind=r.get("kind")
            if field not in ("question","evidence") or not isinstance(quote,str) or not 1<=len(quote)<=400 or quote not in question[field] or kind not in allowed[fam]:
                audit.append({"family":fam,"reason":"source_or_kind_not_bound"});validated.append({"kind":kind,"expression":None,"binding_invalid":True});continue
            item={"kind":kind,"source":field,"quote":quote,"start":question[field].find(quote),
                  "expression":bind_expression(r.get("expression"),schema["column_ids"],visible),"binding_status":"source_bound_model_proposal_not_verified_semantic_truth"}
            if kind=="limit":
                value=r.get("value");item["value"]=value if type(value)is int and value>=0 else None
            if kind=="order":item["direction"]=r.get("direction") if r.get("direction") in ("asc","desc") else None
            validated.append(item)
        result[fam]=validated
    return result,audit

def walk(v):
    if isinstance(v,dict):
        yield v
        for value in v.values():yield from walk(value)
    elif isinstance(v,list):
        for value in v:yield from walk(value)

def lit(v):
    if isinstance(v,dict):
        if v.get("op")!="literal":return None
        value=v.get("value");typ=v.get("literal_type")
    else:value=v;typ=None
    if value is None:return ("null",)
    if typ=="number" or type(value) in (int,float):
        try:return ("number",str(Decimal(str(value)).normalize()))
        except InvalidOperation:return None
    if type(value)is bool:return ("bool",value)
    return ("string",str(value))

def column_key(x):
    if not isinstance(x,dict) or x.get("op")!="column":return None
    return x.get("table"),x.get("column")

def shape(x,population=False,template=None):
    if not isinstance(x,dict):return None
    op=x.get("op")
    if op=="column":return ("column",x.get("table"),x.get("column"))
    if op=="literal":return lit(x)
    if op=="star":return ("star",)
    if op=="not":
        inner=x.get("args",[None])[0]
        inverse={"eq":"neq","neq":"eq","gt":"lte","gte":"lt","lt":"gte","lte":"gt"}
        if isinstance(inner,dict) and inner.get("op") in inverse:
            return shape({**inner,"op":inverse[inner["op"]]})
    if op not in {"count","sum","avg","min","max","add","subtract","multiply","divide","eq","neq","gt","gte","lt","lte","and","or","not"}:return None
    if op in ("and","or"):
        flattened=[]
        def flatten(node):
            if node.get("op")==op:
                for child in node.get("args",[]):flatten(child)
            else:flattened.append(node)
        flatten(x)
        args=[shape(a) for a in flattened]
        if any(a is None for a in args):return None
        return (op,tuple(sorted(args,key=repr)))
    args=[shape(a,population,template=t) for a,t in zip(x.get("args",[]),(template or {}).get("args",x.get("args",[])))]
    if any(a is None for a in args):return None
    if op in {"eq","neq","gt","gte","lt","lte"} and len(args)==2 and repr(args[0])>repr(args[1]):
        args=list(reversed(args));op={"gt":"lt","gte":"lte","lt":"gt","lte":"gte"}.get(op,op)
    if op in {"add","multiply","eq","neq"}:args=sorted(args,key=repr)
    values=[op,tuple(args)]
    if op in {"count","sum","avg","min","max"}:
        t=template or x;distinct=t.get("distinct")
        values.append(bool(x.get("distinct")) if distinct is not None else "unspecified_distinct")
        if t.get("grain") is not None:
            grain=x.get("grain",[])
            keys=[shape(a) for a in grain]
            if any(a is None for a in keys):return None
            values.append(("grain",tuple(sorted(keys,key=repr))))
        if t.get("check_population"):
            pf=x.get("population",{}).get("filter") if "population" in x else x.get("population_filter")
            f=shape(pf) if pf else ("unrestricted",)
            if f is None:return None
            values.append(("population",f))
    return tuple(values)

def equal_expr(expected,actual):
    if expected is None or not isinstance(actual,dict):return None
    aggregate_ops={"count","sum","avg","min","max"}
    aggregates=[x for x in walk(expected) if x.get("op") in aggregate_ops]
    population=bool(aggregates) and all(x.get("check_population") for x in aggregates)
    grain=bool(aggregates) and all(x.get("grain") is not None for x in aggregates)
    distinct=bool(aggregates) and all(x.get("distinct") is not None for x in aggregates)
    def tuple_json(x):return json.dumps(x,sort_keys=True,separators=(",",":"))
    def atom(x):
        op=x.get("op")
        if op in aggregate_ops:
            args=[atom(a) for a in x.get("args",[])]
            if any(a is None for a in args):return None
            out=[op,args]
            if distinct:out.append(bool(x.get("distinct")))
            if grain:
                gg=[shape(a) for a in x.get("grain",[])]
                if any(a is None for a in gg):return None
                out.append(sorted(gg,key=repr))
            if population:
                pf=x.get("population",{}).get("filter") if "population" in x else x.get("population_filter")
                pf=shape(pf) if pf else ("unrestricted",)
                if pf is None:return None
                out.append(pf)
            return tuple_json(out)
        return shape(x)
    def add(a,b,sign=1):
        out=dict(a)
        for k,v in b.items():
            out[k]=out.get(k,Fraction(0))+sign*v
            if not out[k]:del out[k]
        if len(out)>128:raise ValueError("polynomial bound")
        return out
    def mul(a,b):
        out={}
        for ka,va in a.items():
            for kb,vb in b.items():
                k=tuple(sorted(ka+kb));out[k]=out.get(k,Fraction(0))+va*vb
        if len(out)>128:raise ValueError("polynomial bound")
        return {k:v for k,v in out.items() if v}
    def rational(x):
        op=x.get("op")
        if op=="literal":
            value=lit(x)
            if value and value[0]=="number":return {():Fraction(value[1])},{():Fraction(1)}
        if op in ("add","subtract","multiply","divide"):
            aa=x.get("args",[])
            if len(aa)!=2:raise ValueError("binary expression")
            an,ad=rational(aa[0]);bn,bd=rational(aa[1])
            if op=="add":return add(mul(an,bd),mul(bn,ad)),mul(ad,bd)
            if op=="subtract":return add(mul(an,bd),mul(bn,ad),-1),mul(ad,bd)
            if op=="multiply":return mul(an,bn),mul(ad,bd)
            return mul(an,bd),mul(ad,bn)
        key=atom(x)
        if key is None:raise ValueError("unknown expression")
        return {(tuple_json(key),):Fraction(1)},{():Fraction(1)}
    arithmetic={"add","subtract","multiply","divide"}|aggregate_ops|{"column","literal","star"}
    if expected.get("op") in arithmetic and actual.get("op") in arithmetic:
        try:
            en,ed=rational(expected);an,ad=rational(actual)
            if not ed or not ad:return None
            same=mul(en,ad)==mul(an,ed)
            if not same:
                expected_count_star=any(n.get("op")=="count" and any(a.get("op")=="star" for a in n.get("args",[])) for n in walk(expected))
                actual_count_star=any(n.get("op")=="count" and any(a.get("op")=="star" for a in n.get("args",[])) for n in walk(actual))
                if expected_count_star!=actual_count_star:return None
            if same and any(n.get("op")=="divide" and n.get("division_semantics")=="integer_possible" for n in walk(actual)):
                return None
            return same
        except (ValueError,TypeError,KeyError,ZeroDivisionError,InvalidOperation):return None
    a=shape(expected,template=expected);b=shape(actual,template=expected)
    return None if a is None or b is None else a==b

def compare_requirement(r,doc):
    kind=r.get("kind");expected=r.get("expression")
    root=next((s for s in doc.get("scopes",[]) if s["id"]==doc.get("root_scope")),{})
    if doc.get("status")!="parsed":return "unknown","sql_not_supported"
    if any(n.get("op")=="column" for n in walk(expected)):
        instances=Counter(s.get("table") for s in root.get("sources",[]) if not s.get("derived"))
        if any(n.get("op")=="column" and instances[n.get("table")]>1 for n in walk(expected)):
            return "unknown","repeated_table_role_unresolved"
    derived=any(s.get("derived") for s in root.get("sources",[]))
    if kind=="limit":
        target=r.get("value");actual=lit(doc.get("limit"))
        if target is None:return "unknown","limit_unbound"
        if actual is None:return ("unknown","derived_limit_unresolved") if derived else ("contradicted","explicit_limit_missing")
        return ("supported","limit_matches") if actual==lit(target) else ("contradicted","limit_differs")
    if expected is None:return "unknown","requirement_binding_unresolved"
    if kind=="order":
        orders=doc.get("orders",[])
        if not orders:return ("unknown","derived_order_unresolved") if derived else ("contradicted","explicit_order_missing")
        same=equal_expr(expected,orders[0].get("args",[None])[0])
        if same is None or r.get("direction") is None:return "unknown","order_expression_unresolved"
        return ("supported","order_matches") if same and r["direction"]==orders[0].get("direction") else ("contradicted","order_key_or_direction_differs")
    if kind in ("count","ratio","output"):
        actual=doc.get("outputs",[])
        comparisons=[equal_expr(expected,x) for x in actual]
        if any(v is True for v in comparisons):return "supported","output_expression_matches"
        if comparisons and all(v is False for v in comparisons) and not derived:return "contradicted","requested_output_expression_missing_or_differs"
        return "unknown","output_lineage_unresolved"
    if kind in ("filter","relation"):
        actual=doc.get("filter")
        if kind=="relation":
            joins=[n for j in doc.get("scopes",[]) if j["id"]==doc.get("root_scope") for n in j.get("joins",[]) if n.get("join_type")=="inner"]
            candidates=[a for j in joins for a in j.get("args",[])]
            if actual is not None:candidates.append(actual)
        else:candidates=[actual] if actual is not None else []
        keys=[]
        for x in candidates:
            keys.append(x)
            if x and x.get("op")=="and":keys.extend(x.get("args",[]))
        if any(equal_expr(expected,x) is True for x in keys):return "supported","answer_constraint_present"
        if expected.get("op")=="and" and all(any(equal_expr(e,x) is True for x in keys) for e in expected["args"]):
            return "supported","answer_conjunction_present"
        if derived or any(n.get("op") in ("unknown","or","not","exists","subquery") for x in candidates for n in walk(x)):
            return "unknown","answer_scope_or_boolean_entailment_unresolved"
        if kind=="relation":return "unknown","relationship_path_unresolved"
        return "contradicted","explicit_answer_filter_missing_or_differs"
    return "unknown","unsupported_requirement_kind"

def observe(sql,schema,requirements):
    doc=analyze_candidate(sql,{"tables":schema["tables"],"foreign_keys":schema["foreign_keys"]})
    features=[];ledger=[]
    for fam in FAMILIES:
        counts=Counter();entries=requirements.get(fam)
        if doc.get("status")!="parsed":counts["unknown"]+=1
        elif entries is None:counts["unknown"]+=1
        elif entries:
            for i,r in enumerate(entries):
                state,reason=compare_requirement(r,doc);counts[state]+=1
                ledger.append({"family":fam,"requirement":i,"state":state,"reason":reason})
        else:
            ops={n.get("op") for n in walk(doc.get("outputs",[]))}
            present=(fam=="count" and "count" in ops or fam=="ratio" and "divide" in ops
                     or fam=="selection" and bool(doc.get("orders") or doc.get("limit"))
                     or fam=="predicate" and bool(doc.get("filter") or any(s.get("joins") for s in doc.get("scopes",[]))))
            if present:counts["unknown"]+=1
        features.extend(float(counts[s]) for s in ("supported","contradicted","unknown"))
    return features,{"status":doc["status"],"flags":doc.get("flags",[]),"observations":ledger,
                     "capabilities":"partial; self-join roles, arbitrary equivalent rewrites and complex boolean implication remain unknown"}
