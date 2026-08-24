#!/usr/bin/env python3
"""Paper 3 key experiment: can QDQ format keep its speed AND recover accuracy with
better calibration? Re-quantize the 'final' exit with 1024/2048 calibration samples
and MinMax vs Percentile, measure accuracy+latency. Target: match QOperator accuracy
(within ~0.6pt of fp32) while keeping the QDQ speedup (2.5x)."""
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
    t=[time.perf_counter() for _ in [0]]
    tt=[]
    for _ in range(it): t0=time.perf_counter(); s.run(None,{i:x}); tt.append((time.perf_counter()-t0)*1000)
    return float(np.mean(tt))

EX="final"; fp32=MODELS/f"mnv3_c100_{EX}_fp32.onnx"
iname=ort.InferenceSession(str(fp32),providers=["CPUExecutionProvider"]).get_inputs()[0].name
fpm=lat(sess(fp32)); fa=acc(sess(fp32))
print(f"Paper 3 '{EX}' QDQ calibration sweep (target: fp32 acc={fa:.2f}%, {fpm:.3f}ms)\n")
print(f"{'config':34s} {'lat':>8s} {'acc':>7s} {'speedup':>8s} {'acc_drop':>9s}")
print("-"*70)
configs=[
 ("QDQ minmax 256",256,CalibrationMethod.MinMax,True),
 ("QDQ minmax 1024",1024,CalibrationMethod.MinMax,True),
 ("QDQ minmax 2048",2048,CalibrationMethod.MinMax,True),
 ("QDQ percentile 1024",1024,CalibrationMethod.Percentile,True),
 ("QDQ percentile 2048",2048,CalibrationMethod.Percentile,True),
]
res=[{"fp32":{"lat":fpm,"acc":fa}}]
for name,n,method,pc in configs:
    out=MODELS/f"mnv3_c100_{EX}_qdq_test.onnx"
    try:
        quantize_static(str(fp32),str(out),Reader(iname,n),quant_format=QuantFormat.QDQ,
                        per_channel=pc,weight_type=QuantType.QInt8,activation_type=QuantType.QInt8,
                        calibrate_method=method)
        s=sess(out); lm=lat(s); a=acc(s); del s; gc.collect()
        print(f"{name:34s} {lm:7.3f}ms {a:6.2f}% {fpm/lm:7.2f}x {fa-a:+8.2f}pt")
        res.append({"config":name,"lat":lm,"acc":a,"speedup":fpm/lm,"acc_drop":fa-a})
    except Exception as e:
        print(f"{name:34s} ERR {str(e)[:40]}")
json.dump(res,open(Path.home()/"paper3_qdq_calib.json","w"),indent=2)
print("\nGoal: a QDQ config with speedup>2x AND acc_drop<1pt = the clean Paper 3 result.")
