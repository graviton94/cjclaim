import json, statistics
p='artifacts/trained_models_v3.json'
with open(p,'r',encoding='utf-8-sig') as f:
    data=json.load(f)
if isinstance(data,dict) and 'series_models' in data:
    data = data['series_models']
summary={'total':len(data)}
# counters
counters={
    'n_train_ge_18':0,'n_train_ge_24':0,
    'nonzero_ge_5':0,'nonzero_ge_10':0,
    'train_var_pos':0,'max_zero_run_lt_12':0,
    'has_aic':0,'has_arroots':0,'min_abs_root_gt_1.005':0,
    'has_all_fields':0
}
missing_field_counts={}
min_abs_vals=[]
for k,v in data.items():
    if not isinstance(v,dict):
        missing_field_counts.setdefault('not_dict',0)
        missing_field_counts['not_dict']+=1
        continue
    n=v.get('n_train_points')
    if isinstance(n,int):
        if n>=18: counters['n_train_ge_18']+=1
        if n>=24: counters['n_train_ge_24']+=1
    else:
        missing_field_counts.setdefault('n_train_missing',0)
        missing_field_counts['n_train_missing']+=1
    nz=v.get('nonzero_pct')
    if isinstance(nz,(int,float)):
        if nz>=5.0: counters['nonzero_ge_5']+=1
        if nz>=10.0: counters['nonzero_ge_10']+=1
    else:
        missing_field_counts.setdefault('nonzero_missing',0)
        missing_field_counts['nonzero_missing']+=1
    tv=v.get('train_var')
    if isinstance(tv,(int,float)) and tv>0:
        counters['train_var_pos']+=1
    else:
        missing_field_counts.setdefault('train_var_nonpos_or_missing',0)
        missing_field_counts['train_var_nonpos_or_missing']+=1
    mz=v.get('max_zero_run')
    if isinstance(mz,int) and mz<12:
        counters['max_zero_run_lt_12']+=1
    else:
        missing_field_counts.setdefault('max_zero_missing_or_ge12',0)
        missing_field_counts['max_zero_missing_or_ge12']+=1
    aic=v.get('aic')
    if aic is not None:
        counters['has_aic']+=1
    else:
        missing_field_counts.setdefault('aic_missing',0)
        missing_field_counts['aic_missing']+=1
    ar=v.get('arroots')
    min_abs=None
    if ar is not None:
        if isinstance(ar,(list,tuple)):
            try:
                absvals=[abs(float(x)) for x in ar if x is not None]
                if absvals:
                    min_abs=min(absvals)
            except Exception:
                pass
        elif isinstance(ar,dict):
            if 'min_abs_root' in ar:
                try:
                    min_abs=float(ar['min_abs_root'])
                except Exception:
                    pass
            else:
                r=ar.get('roots') or ar.get('values')
                if isinstance(r,(list,tuple)):
                    try:
                        absvals=[abs(float(x)) for x in r if x is not None]
                        if absvals:
                            min_abs=min(absvals)
                    except Exception:
                        pass
        elif isinstance(ar,(int,float)):
            min_abs=abs(float(ar))
        if min_abs is not None:
            counters['has_arroots']+=1
            min_abs_vals.append(min_abs)
            if min_abs>1.005:
                counters['min_abs_root_gt_1.005']+=1
        else:
            missing_field_counts.setdefault('arroots_unparsable',0)
            missing_field_counts['arroots_unparsable']+=1
    else:
        missing_field_counts.setdefault('arroots_missing',0)
        missing_field_counts['arroots_missing']+=1
    required=['aic','arroots','n_train_points','nonzero_pct','train_var','max_zero_run']
    if all((k in v and v.get(k) is not None) for k in required):
        counters['has_all_fields']+=1

if min_abs_vals:
    min_min=min(min_abs_vals)
    max_min=max(min_abs_vals)
    median=statistics.median(min_abs_vals)
else:
    min_min=max_min=median=None

out={'summary':summary,'counters':counters,'min_abs_stats':{'min':min_min,'median':median,'max':max_min},'missing_field_counts':missing_field_counts}
print(json.dumps(out,ensure_ascii=False,indent=2))
