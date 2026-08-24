#!/usr/bin/env python3
"""Definitive Paper 3 dataset: all 3 exits x {fp32, QOperator, QDQ-correct}.
QDQ re-exported with CORRECT 0-1 calibration (MinMax 256). accuracy + latency."""
import numpy as np, tarfile, pickle, time, gc, json
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader, CalibrationMethod
from pathlib import Path
MODELS=Path.home()/"tier1-experiments/models"; DATA=Path.home()/"Desktop/researchpaper3/data"
with tarfile.open(DATA/"cifar-100-python.tar.gz","r:gz") as t:
    d=pickle.loads(t.extractfile("cifar-100-python/test").read(),encoding="latin1")
raw=np.asarray(d["data"],dtype=np.uint8); y=np.array(d["fine_labels"],dtype=np.int64); N=len(y)
def prep(r): return r.reshape(-1,3,32,32).astype(np.float32)/255.0
class Reader(CalibrationDataReader):
    def __init__(self,iname,n): self.it=iter([{iname:prep(raw[i:i+1])} for i in range(n)])
    def get_next(self): return next(self.it,None)
def sess(p):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(p),sess_options=so,providers=["CPUExecutionProvider"])
def acc(s,bs=200):
    i=s.get_inputs()[0].name; c=0
    for k in range(0,N,bs): c+=int((s.run(None,{i:prep(raw[k:k+bs])})[0].argmax(1)==y[k:k+bs]).sum())
    return 100*c/N
def lat(s,it=300,wu=50):
    i=s.get_inputs()[0].name; x=prep(raw[0:1])
    for _ in range(wu): s.run(None,{i:x})
    tt=[]
    for _ in range(it): t0=time.perf_counter(); s.run(None,{i:x}); tt.append((time.perf_counter()-t0)*1000)
    return float(np.mean(tt)),float(np.percentile(tt,50))

print("DEFINITIVE Paper 3 dataset (QDQ re-calibrated with correct 0-1 preprocessing)\n")
print(f"{'exit':6s} {'variant':16s} {'lat':>8s} {'p50':>8s} {'acc':>7s} {'speedup':>8s}")
print("-"*60)
out=[]
for ex in ["exit1","exit2","final"]:
    fp32=MODELS/f"mnv3_c100_{ex}_fp32.onnx"
    iname=ort.InferenceSession(str(fp32),providers=["CPUExecutionProvider"]).get_inputs()[0].name
    qdq=MODELS/f"mnv3_c100_{ex}_qdq_clean.onnx"
    quantize_static(str(fp32),str(qdq),Reader(iname,256),quant_format=QuantFormat.QDQ,
                    per_channel=True,weight_type=QuantType.QInt8,activation_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    row={"exit":ex}
    for var,p in [("fp32",fp32),("int8-QOperator",MODELS/f"mnv3_c100_{ex}_int8.onnx"),("int8-QDQ-clean",qdq)]:
        s=sess(p); lm,lp=lat(s); a=acc(s); del s; gc.collect()
        base=row.get("fp32",{}).get("lat_mean",lm)
        row[var]={"lat_mean":lm,"lat_p50":lp,"acc":a,"speedup":base/lm}
        print(f"{ex:6s} {var:16s} {lm:7.3f}ms {lp:7.3f}ms {a:6.2f}% {base/lm:7.2f}x")
    out.append(row); print()
json.dump(out,open(Path.home()/"paper3_clean_dataset.json","w"),indent=2)
print("Saved ~/paper3_clean_dataset.json  -- this is the definitive Paper 3 table.")
