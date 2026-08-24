#!/usr/bin/env python3
"""Does sustained early-exit inference heat the Pi 5 enough for a thermal cap to
bind? Run each exit continuously on 4 threads for 45s, watch temp + throughput.
Determines whether thermal-aware exit selection has anything to control."""
import numpy as np, time, threading, subprocess
import onnxruntime as ort
from pathlib import Path
M=Path.home()/"tier1-experiments/models"

def temp():
    try: return int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000
    except: return -1

def prep(): return np.random.randn(1,3,32,32).astype(np.float32)

def sustained(model_path, secs=45, nthreads=4):
    so=ort.SessionOptions(); so.intra_op_num_threads=1  # 1 per worker thread
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess=ort.InferenceSession(str(model_path),sess_options=so,providers=["CPUExecutionProvider"])
    iname=sess.get_inputs()[0].name
    stop=threading.Event(); counts=[0]*nthreads
    def worker(wid):
        x=prep()
        while not stop.is_set():
            sess.run(None,{iname:x}); counts[wid]+=1
    ts=[threading.Thread(target=worker,args=(w,),daemon=True) for w in range(nthreads)]
    t0=time.time(); temps=[]
    for t in ts: t.start()
    while time.time()-t0<secs:
        time.sleep(3); temps.append(temp())
    stop.set(); time.sleep(0.5)
    thr=sum(counts)/(time.time()-t0)
    return temps, thr

print(f"start temp {temp():.1f}C\n")
print(f"{'exit':8s} {'start':>6s} {'peak':>6s} {'end':>6s} {'throughput':>12s}")
print("-"*46)
for ex in ["exit1","exit2","final"]:
    t_start=temp()
    temps,thr=sustained(M/f"mnv3_c100_{ex}_fp32.onnx")
    print(f"{ex:8s} {t_start:5.1f}C {max(temps):5.1f}C {temps[-1]:5.1f}C {thr:9.0f} inf/s")
    # cool down between
    time.sleep(20)
print(f"\ncoupling law predicts T=45.6+0.175*U; at U=400% (4 cores) T~=115C but")
print("throttle/heat-spreader caps it. If peak temp < ~60C, single-model load does not")
print("bind a realistic cap -> the controller needs multi-tenant or aggressive-cap framing.")
