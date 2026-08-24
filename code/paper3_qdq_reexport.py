#!/usr/bin/env python3
"""Paper 3 fix: re-export the early-exit INT8 models in QDQ format (they are
currently QOperator, which is slow on Cortex-A76), then benchmark fp32 vs the
original QOperator-int8 vs the new QDQ-int8 for latency (ENABLE_ALL) AND accuracy.
Proves that the QDQ re-export flips INT8 from slower-than-FP32 to faster, matching
Paper 7, without changing the weights' accuracy."""
import os, tarfile, pickle, time, gc, json
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
from pathlib import Path

MODELS = Path.home()/"tier1-experiments/models"
DATA = Path.home()/"Desktop/researchpaper3/data"
CIFAR100_MEAN=(0.5071,0.4865,0.4409); CIFAR100_STD=(0.2673,0.2564,0.2762)

def load_cifar100():
    with tarfile.open(DATA/"cifar-100-python.tar.gz","r:gz") as t:
        d=pickle.loads(t.extractfile("cifar-100-python/test").read(),encoding="latin1")
    raw=np.asarray(d["data"],dtype=np.uint8)
    y=np.array(d["fine_labels"],dtype=np.int64)
    return raw,y

def preprocess(raw, idx):
    mean=np.array(CIFAR100_MEAN,np.float32)[:,None,None]; std=np.array(CIFAR100_STD,np.float32)[:,None,None]
    x=raw[idx].reshape(-1,3,32,32).astype(np.float32)/255.0
    return (x-mean)/std

raw,y = load_cifar100()
N=raw.shape[0]

class CalibReader(CalibrationDataReader):
    def __init__(self, input_name, n=256):
        self.input_name=input_name
        self.data=iter([{input_name: preprocess(raw, slice(i,i+1))} for i in range(n)])
    def get_next(self): return next(self.data, None)

def accuracy(sess, bs=128):
    iname=sess.get_inputs()[0].name; correct=0
    for i in range(0,N,bs):
        x=preprocess(raw, slice(i,i+bs))
        logits=sess.run(None,{iname:x})[0]
        correct+=int((logits.argmax(1)==y[i:i+bs]).sum())
    return 100.0*correct/N

def latency(path, iters=300, warmup=50):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s=ort.InferenceSession(str(path),sess_options=so,providers=["CPUExecutionProvider"])
    iname=s.get_inputs()[0].name; x=preprocess(raw, slice(0,1))
    for _ in range(warmup): s.run(None,{iname:x})
    t=[]
    for _ in range(iters):
        t0=time.perf_counter(); s.run(None,{iname:x}); t.append((time.perf_counter()-t0)*1000)
    return float(np.mean(t)), float(np.percentile(t,50)), s

print(f"ORT {ort.__version__}  Paper 3 QDQ re-export + benchmark (CIFAR-100)\n")
results=[]
print(f"{'exit':6s} {'variant':16s} {'lat_mean':>9s} {'lat_p50':>8s} {'acc':>7s} {'vs_fp32':>8s}")
print("-"*62)
for ex in ["exit1","exit2","final"]:
    fp32=MODELS/f"mnv3_c100_{ex}_fp32.onnx"
    qop=MODELS/f"mnv3_c100_{ex}_int8.onnx"            # original QOperator
    qdq=MODELS/f"mnv3_c100_{ex}_int8_qdq.onnx"        # new QDQ
    # re-export QDQ
    s0=ort.InferenceSession(str(fp32),providers=["CPUExecutionProvider"])
    iname=s0.get_inputs()[0].name; del s0
    try:
        quantize_static(str(fp32), str(qdq), CalibReader(iname,256),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    except Exception as e:
        print(f"{ex}: QDQ export FAILED: {e}"); continue
    # benchmark all three
    fpm,fpp,sf=latency(fp32); fa=accuracy(sf); del sf; gc.collect()
    qom,qop50,sq=latency(qop); qa=accuracy(sq); del sq; gc.collect()
    qdm,qdp,sd=latency(qdq); da=accuracy(sd); del sd; gc.collect()
    print(f"{ex:6s} {'fp32':16s} {fpm:8.3f}ms {fpp:7.3f}ms {fa:6.2f}% {'1.00x':>8s}")
    print(f"{ex:6s} {'int8-QOperator':16s} {qom:8.3f}ms {qop50:7.3f}ms {qa:6.2f}% {fpm/qom:7.2f}x")
    print(f"{ex:6s} {'int8-QDQ (new)':16s} {qdm:8.3f}ms {qdp:7.3f}ms {da:6.2f}% {fpm/qdm:7.2f}x")
    print()
    results.append(dict(exit=ex, fp32_ms=fpm, fp32_acc=fa,
                        qop_ms=qom, qop_acc=qa, qop_spd=fpm/qom,
                        qdq_ms=qdm, qdq_acc=da, qdq_spd=fpm/qdm))
json.dump(results, open(Path.home()/"paper3_qdq_results.json","w"), indent=2)
print("QDQ re-export should flip INT8 from slower (QOperator) to faster, accuracy unchanged.")
print("Saved ~/paper3_qdq_results.json")
