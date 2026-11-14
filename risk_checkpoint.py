"""SQL adaptation of LearnRisk distributions with a frozen neural prior."""
from __future__ import annotations
import hashlib,json,math
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONFIG={"seed":13,"epochs":200,"learning_rate":0.03,"alpha":0.9,"bce_weight":1.0,"boundary_weight":1.0,
        "revision":"question_contrast_rules_fixed_prior_bounded_mass","max_rule_mass":0.15,"same_question_margin":0.02,
        "rule_initial_unnormalized_mass":0.1,"weight_regularization":0.001,
        "boundary_temperature":0.1,"regularization":0.01,"checkpoint":"fixed_last_epoch",
        "rule_support_unit":"distinct_questions","conditional_prior_bins":10,"prior_pseudocount":20,
        "minimum_mixed_varying_rule_questions":3,"std_scale_bounds":[0.5,2.0],
        "correlation_policy":"duplicate_activations_share_parameters; fully_correlated_std_aggregation"}
def sigmoid_np(x):return 1/(1+np.exp(-np.clip(x,-700,700)))
def sql_weights(qids,sql_hashes):
    rows=defaultdict(lambda:defaultdict(list))
    for i,(q,s) in enumerate(zip(qids,sql_hashes)):rows[int(q)][s].append(i)
    weights=np.zeros(len(qids),dtype=np.float64)
    for groups in rows.values():
        for indices in groups.values():weights[indices]=1/(len(groups)*len(indices))
    return weights

def build_bank(activation,pbase,wrong,qids,sql_hashes,forest):
    n,r=activation.shape;weights=sql_weights(qids,sql_hashes);bins=np.minimum((pbase*10).astype(int),9)
    buckets=[]
    for b in range(10):
        mask=bins==b;w=weights[mask];y=wrong[mask];nq=len(set(np.asarray(qids)[mask].tolist()))
        total=w.sum();mean=(float(np.dot(w,y))+1)/(float(total)+2)
        std=math.sqrt(mean*(1-mean)/(nq+3))
        buckets.append({"mean":mean,"std":max(.005,std),"questions":nq})
    grouped={};groups=[]
    for j in range(r):
        key=hashlib.sha256(np.packbits(activation[:,j]).tobytes()).hexdigest()
        if key not in grouped:grouped[key]=len(groups);groups.append({"members":[],"activation_sha256":key})
        groups[grouped[key]]["members"].append(j)
    means=[];stds=[];reliability=[];details=[]
    byq=defaultdict(list)
    for i,q in enumerate(qids):byq[int(q)].append(i)
    for group in groups:
        a=activation[:,group["members"][0]].astype(bool);directions=[]
        for indices in byq.values():
            idx=np.asarray(indices);hit=a[idx]
            if not hit.any() or hit.all() or not 0<wrong[idx].sum()<len(idx):continue
            wh=weights[idx][hit];wn=weights[idx][~hit]
            directions.append(float(np.dot(wh,wrong[idx][hit])/wh.sum()-np.dot(wn,wrong[idx][~hit])/wn.sum()))
        direction=float(np.mean(directions)) if directions else 0.
        classes={forest["rules"][j]["predicted_correct"] for j in group["members"]}
        consistent=(classes=={1} and direction<0 or classes=={0} and direction>0)
        rel=min(1.,len(directions)/20.) if consistent and len(directions)>=3 else 0.
        gm=[];gs=[];supports=[]
        for b in range(10):
            hit=a&(bins==b);w=weights[hit];yy=wrong[hit]
            nq=len(set(np.asarray(qids)[hit].tolist()))
            # Conditional true error frequency; never insert a ranking gain as a probability.
            prior=buckets[b]["mean"];total=float(w.sum())
            mean=(float(np.dot(w,yy))+20*prior)/(total+20)
            std=math.sqrt(mean*(1-mean)/(nq+21))
            gm.append(mean);gs.append(max(.005,std));supports.append(nq)
        means.append(gm);stds.append(gs);reliability.append(rel)
        details.append({"members":group["members"],"mixed_varying_questions":len(directions),"within_question_wrong_rate_difference":direction,
                        "direction_consistent_with_original_rule":bool(consistent),"conditional_question_support":supports,"reliability":rel})
    return {"means":means,"stds":stds,"reliability":reliability,"groups":groups,"rule_audit":details,
            "prior_bins":buckets,"conditional_means_are_empirical_probabilities":True,"discovery_only":True,
            "config":CONFIG,"rule_count":r,"group_count":len(groups)}

def group_activation(a,bank):
    n=len(a);out=np.zeros((n,len(bank["groups"])),dtype=np.float64)
    for g,info in enumerate(bank["groups"]):out[:,g]=np.max(a[:,info["members"]],axis=1)
    return out

def truncated(location,std,alpha=.9):
    s=std.clamp_min(1e-5);z0=-location/s;z1=(1-location)/s
    low=torch.special.ndtr(z0);mass=(torch.special.ndtr(z1)-low).clamp_min(1e-12)
    phi=lambda z:torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    mean=location+s*(phi(z0)-phi(z1))/mass
    probability=(low+alpha*mass).clamp(1e-15,1-1e-15)
    var=location+s*torch.special.ndtri(probability)
    return mean.clamp(1e-12,1-1e-12),var.clamp(0,1-1e-12)

class PriorRuleVaR(nn.Module):
    def __init__(self,bank):
        super().__init__();self.bank=bank;g=bank["group_count"]
        self.register_buffer("means",torch.tensor(bank["means"],dtype=torch.float64).reshape(g,10))
        self.register_buffer("stds",torch.tensor(bank["stds"],dtype=torch.float64).reshape(g,10))
        self.register_buffer("reliability",torch.tensor(bank["reliability"],dtype=torch.float64))
        self.register_buffer("prior_std",torch.tensor([x["std"] for x in bank["prior_bins"]],dtype=torch.float64))
        # One prior and one weight per distinct discovery activation pattern.
        init=torch.zeros(g+1,dtype=torch.float64)
        if g:init[1:]=math.log(CONFIG["rule_initial_unnormalized_mass"])
        self.register_buffer("initial_weights",init)
        self.log_weights=nn.Parameter(init.clone())
        self.raw_std_scale=nn.Parameter(torch.zeros(g+1,dtype=torch.float64))
        self.raw_prior_slope=nn.Parameter(torch.tensor(math.log(math.expm1(1.)),dtype=torch.float64))
        self.prior_bias=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.raw_prior_slope.requires_grad_(False);self.prior_bias.requires_grad_(False)

    def forward(self,pbase,activation,invalid):
        p=pbase.clamp(1e-8,1-1e-8);bins=torch.clamp((pbase*10).long(),0,9)
        calibrated=pbase
        n=len(p);g=self.bank["group_count"]
        if g:
            mu=torch.cat([calibrated[:,None],self.means[:,bins].T],dim=1)
            sd=torch.cat([self.prior_std[bins,None],self.stds[:,bins].T],dim=1)
            rel=torch.cat([torch.ones((n,1),dtype=p.dtype),activation*self.reliability[None,:]],dim=1)
        else:
            mu=calibrated[:,None];sd=self.prior_std[bins,None];rel=torch.ones((n,1),dtype=p.dtype)
        mass=self.log_weights[None,:]+rel.clamp_min(1e-300).log()
        mass=mass.masked_fill(rel<=0,-torch.inf);w=torch.softmax(mass,dim=1)
        # Scale interval [0.5,2], initialized at exactly 1.
        if g:
            rule_mass=w[:,1:].sum(1,keepdim=True)
            capped=rule_mass.clamp_max(CONFIG["max_rule_mass"])
            w=torch.cat([1-capped,w[:,1:]*capped/rule_mass.clamp_min(1e-30)],dim=1)
        scale=.5+1.5*torch.sigmoid(self.raw_std_scale-math.log(2.))
        sd=sd.clone();sd[:,0]=0
        location=(w*mu).sum(1);std=(w*sd*scale[None,:]).sum(1)
        mean,risk=truncated(location,std,CONFIG["alpha"])
        active=(rel[:,1:].sum(1)>0) if g else torch.zeros(n,dtype=torch.bool)
        mean=torch.where(active,mean,pbase);risk=torch.where(active,risk,pbase)
        return torch.where(invalid,torch.ones_like(mean),mean),torch.where(invalid,torch.ones_like(risk),risk)

    def penalty(self):
        return CONFIG["weight_regularization"]*(self.log_weights-self.initial_weights).square().mean()+CONFIG["regularization"]*(
            self.raw_std_scale.square().mean()+self.prior_bias.square()+(F.softplus(self.raw_prior_slope)-1).square())

    def payload(self):
        return {"artifact_type":"prior_conditional_original_rule_var_sql_adaptation_v1","bank":self.bank,"config":CONFIG,
                "state":{k:v.detach().cpu().tolist() for k,v in self.state_dict().items()},"contains_optimizer_state":False}

def restore(payload):
    if payload["artifact_type"]!="prior_conditional_original_rule_var_sql_adaptation_v1" or payload["config"]!=CONFIG:raise ValueError("risk model contract drift")
    m=PriorRuleVaR(payload["bank"]);template=m.state_dict();raw=payload["state"]
    if set(template)!=set(raw):raise ValueError("state fields mismatch")
    learnable=set(dict(m.named_parameters()))
    state={k:torch.tensor(raw[k],dtype=v.dtype).reshape(v.shape) for k,v in template.items()}
    for k in template:
        if k not in learnable and not torch.equal(template[k],state[k]):raise ValueError("fixed bank changed")
    m.load_state_dict(state,strict=True);return m

def fit(bank,pbase,activation,wrong,qids,sql_hashes,invalid,progress,checkpoint=None):
    torch.set_num_threads(2);torch.manual_seed(13);torch.use_deterministic_algorithms(True)
    model=PriorRuleVaR(bank);opt=torch.optim.Adam(model.parameters(),lr=CONFIG["learning_rate"])
    p=torch.tensor(pbase,dtype=torch.float64);a=torch.tensor(activation,dtype=torch.float64)
    y=torch.tensor(wrong,dtype=torch.float64);bad=torch.tensor(invalid,dtype=torch.bool)
    qsorted=sorted(set(map(int,qids)));qi={q:i for i,q in enumerate(qsorted)}
    qindex=torch.tensor([qi[int(q)] for q in qids],dtype=torch.long)
    weights=torch.tensor(sql_weights(qids,sql_hashes),dtype=torch.float64)/len(qsorted)
    mixed=torch.tensor([bool(0<sum(wrong[np.asarray(qids)==q])<sum(np.asarray(qids)==q)) for q in qsorted],dtype=torch.bool)
    history=[]
    start_epoch=1
    if checkpoint is not None:
        from pathlib import Path
        checkpoint=Path(checkpoint)
        if checkpoint.exists():
            saved=torch.load(checkpoint,weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);history=saved['history'];start_epoch=saved['epoch']+1
    for epoch in range(start_epoch,CONFIG["epochs"]+1):
        opt.zero_grad(set_to_none=True);mean,risk=model(p,a,bad)
        bce=(F.binary_cross_entropy(mean.clamp(1e-12,1-1e-12),y,reduction="none")*weights).sum()
        good_min=torch.full((len(qsorted),),float("inf"),dtype=torch.float64).scatter_reduce(
            0,qindex,torch.where(y==0,risk,torch.full_like(risk,float("inf"))),reduce="amin",include_self=True)
        bad_min=torch.full((len(qsorted),),float("inf"),dtype=torch.float64).scatter_reduce(
            0,qindex,torch.where(y==1,risk,torch.full_like(risk,float("inf"))),reduce="amin",include_self=True)
        boundary=F.softplus((CONFIG["same_question_margin"]+good_min[mixed]-bad_min[mixed])/CONFIG["boundary_temperature"]).mean() if mixed.any() else risk.sum()*0
        penalty=model.penalty();loss=bce+boundary+penalty
        if not torch.isfinite(loss):raise ValueError("nonfinite full risk loss")
        loss.backward()
        if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in model.parameters()):raise ValueError("nonfinite risk gradient")
        opt.step()
        item={"epoch":epoch,"loss":float(loss.detach()),"bce":float(bce.detach()),"boundary":float(boundary.detach()),"regularization":float(penalty.detach())}
        history.append(item)
        if checkpoint is not None and (epoch%10==0 or epoch==CONFIG['epochs']):
            tmp=checkpoint.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history,epoch=epoch),tmp);tmp.replace(checkpoint)
        if epoch==1 or epoch%10==0:progress("RISK_FIT_PROGRESS",**item)
    return model,history

def predict(model,pbase,activation,invalid):
    means=[];risks=[]
    with torch.no_grad():
        for start in range(0,len(pbase),4096):
            stop=start+4096
            mean,risk=model(torch.tensor(pbase[start:stop],dtype=torch.float64),torch.tensor(activation[start:stop],dtype=torch.float64),torch.tensor(invalid[start:stop],dtype=torch.bool))
            means.append(mean.numpy());risks.append(risk.numpy())
    return np.concatenate(means),np.concatenate(risks)
