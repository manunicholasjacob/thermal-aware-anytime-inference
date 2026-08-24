#!/usr/bin/env python3
"""Test Paper 3's own INT8 early-exit models: do they achieve the INT8 speedup
(like Paper 7's) at ENABLE_ALL, or are they slow regardless (model-level cause)?"""
import numpy as np, time, onnxruntime as ort
from pathlib import Path
M = Path.home()/"tier1-experiments/models"
LEV = {"BASIC":ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
       "ALL":ort.GraphOptimizationLevel.ORT_ENABLE_ALL}

def bench(p, lvl, it=200, wu=40):
    so=ort.SessionOptions(); so.intra_op_num_threads=4; so.graph_optimization_level=lvl
    s=ort.InferenceSession(str(p),sess_options=so,providers=["CPUExecutionProvider"])
    i0=s.get_inputs()[0].name
    sh=[d if isinstance(d,int) else 1 for d in s.get_inputs()[0].shape]
    x=np.random.randn(*sh).astype("float32")
    for _ in range(wu): s.run(None,{i0:x})
    t=[]
    for _ in range(it):
        t0=time.perf_counter(); s.run(None,{i0:x}); t.append((time.perf_counter()-t0)*1000)
    return float(np.mean(t))

def qtype(p):
    """Inspect quantization op types to see QDQ vs QOperator format."""
    import onnx
    m=onnx.load(str(p)); ops=set(n.op_type for n in m.graph.node)
    qdq = {"QuantizeLinear","DequantizeLinear"} & ops
    qop = {o for o in ops if o.startswith("QLinear") or o in ("ConvInteger","MatMulInteger")}
    return ("QDQ" if qdq else "")+("+QOp" if qop else "") or "?", sorted(ops)[:12]

print(f"ORT {ort.__version__}  -- Paper 3 early-exit MobileNetV3 INT8 models\n")
print(f"{'exit':8s} {'fp32':>9s} {'int8@BASIC':>11s} {'int8@ALL':>10s} {'spd@ALL':>8s}  fmt")
print("-"*62)
for ex in ["exit1","exit2","final"]:
    fp=M/f"mnv3_c100_{ex}_fp32.onnx"; iq=M/f"mnv3_c100_{ex}_int8.onnx"
    if not fp.exists() or not iq.exists(): print(f"{ex}: missing"); continue
    f=bench(fp,LEV["ALL"]); ib=bench(iq,LEV["BASIC"]); ia=bench(iq,LEV["ALL"])
    try: fmt,_=qtype(iq)
    except Exception: fmt="?"
    print(f"{ex:8s} {f:8.3f}ms {ib:10.3f}ms {ia:9.3f}ms {f/ia:7.2f}x  {fmt}")
print("\nPaper 3 tab:anytime claimed INT8 SLOWER. If int8@ALL is faster here, Paper 3")
print("mismeasured; if slower even at ALL, its quantization format differs from Paper 7's.")
