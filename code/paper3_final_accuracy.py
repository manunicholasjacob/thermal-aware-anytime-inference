#!/usr/bin/env python3
"""Paper 3 complete dataset: correct accuracy (0-1 scaling) + latency (ENABLE_ALL)
for all exits x {fp32, int8-QOperator, int8-QDQ}. Full CIFAR-100 test set."""
import numpy as np, tarfile, pickle, time, gc, json
import onnxruntime as ort
from pathlib import Path
MODELS=Path.home()/"tier1-experiments/models"; DATA=Path.home()/"Desktop/researchpaper3/data"
with tarfile.open(DATA/"cifar-100-python.tar.gz","r:gz") as t:
    d=pickle.loads(t.extractfile("cifar-100-python/test").read(),encoding="latin1")
raw=np.asarray(d["data"],dtype=np.uint8); y=np.array(d["fine_labels"],dtype=np.int64); N=len(y)
def prep(r): return r.reshape(-1,3,32,32).astype(np.float32)/255.0   # 0-1 scaling (correct)
def sess(p):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(p),sess_options=so,providers=["CPUExecutionProvider"])
def acc(s,bs=200):
    i=s.get_inputs()[0].name; c=0
    for k in range(0,N,bs):
        c+=int((s.run(None,{i:prep(raw[k:k+bs])})[0].argmax(1)==y[k:k+bs]).sum())
    return 100*c/N
def lat(s,it=300,wu=50):
    i=s.get_inputs()[0].name; x=prep(raw[0:1])
    for _ in range(wu): s.run(None,{i:x})
    t=[]
    for _ in range(it): t0=time.perf_counter(); s.run(None,{i:x}); t.append((time.perf_counter()-t0)*1000)
    return float(np.mean(t)),float(np.percentile(t,50))

print("Paper 3 FINAL dataset — correct preprocessing (0-1 scaling)\n")
print(f"{'exit':6s} {'variant':16s} {'lat_mean':>9s} {'p50':>8s} {'acc':>7s} {'speedup':>8s}")
print("-"*60)
res=[]
for ex in ["exit1","exit2","final"]:
    row={"exit":ex}
    for var,fn in [("fp32",f"mnv3_c100_{ex}_fp32.onnx"),
                   ("int8-QOperator",f"mnv3_c100_{ex}_int8.onnx"),
                   ("int8-QDQ",f"mnv3_c100_{ex}_int8_qdq.onnx")]:
        p=MODELS/fn
        if not p.exists(): print(f"{ex} {var}: MISSING"); continue
        s=sess(p); lm,lp=lat(s); a=acc(s); del s; gc.collect()
        row[var]={"lat_mean":lm,"lat_p50":lp,"acc":a}
        sp = row["fp32"]["lat_mean"]/lm if "fp32" in row else 1.0
        print(f"{ex:6s} {var:16s} {lm:8.3f}ms {lp:7.3f}ms {a:6.2f}% {sp:7.2f}x")
    res.append(row); print()
json.dump(res,open(Path.home()/"paper3_final_dataset.json","w"),indent=2)
print("Saved ~/paper3_final_dataset.json")
