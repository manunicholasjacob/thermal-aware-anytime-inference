#!/usr/bin/env python3
"""Paper 3 definitive controller experiment (honest premise).

At a FIXED request rate below saturation, exit depth sets CPU utilization and thus
temperature (deeper exit = more compute/req = hotter). Under a binding thermal cap,
a thermal-aware controller should hold the cap while maximizing REAL accuracy by
choosing the deepest exit that stays thermally safe -- beating static policies that
either violate the cap (always-final) or waste accuracy (always-exit1).

Compares: always-exit1 / always-exit2 / always-final / adaptive-controller.
Real CIFAR-100 accuracy (vs labels). Reports accuracy, thermal violations, p99 latency.
"""
import numpy as np, time, threading, tarfile, pickle, json, queue
import onnxruntime as ort
from pathlib import Path

M=Path.home()/"tier1-experiments/models"; DATA=Path.home()/"Desktop/researchpaper3/data"
with tarfile.open(DATA/"cifar-100-python.tar.gz","r:gz") as t:
    d=pickle.loads(t.extractfile("cifar-100-python/test").read(),encoding="latin1")
RAW=np.asarray(d["data"],dtype=np.uint8); Y=np.array(d["fine_labels"],dtype=np.int64)
def img(i): return (RAW[i:i+1].reshape(-1,3,32,32).astype(np.float32)/255.0)

EXITS=["exit1","exit2","final"]
# per-exit accuracy (measured) and service time (ms, QDQ-clean, ENABLE_ALL)
ACC={"exit1":19.45,"exit2":32.73,"final":33.65}
SVC={"exit1":0.086,"exit2":0.168,"final":0.594}   # ms per inference

def mksess(ex):
    so=ort.SessionOptions(); so.intra_op_num_threads=1
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(M/f"mnv3_c100_{ex}_qdq_clean.onnx"),sess_options=so,
                                providers=["CPUExecutionProvider"])
SESS={ex:mksess(ex) for ex in EXITS}
INAME=SESS["final"].get_inputs()[0].name

def temp():
    try: return int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000
    except: return -1

def run_policy(name, select_fn, rate_hz, cap, dur=60, nworkers=4):
    """select_fn(cur_temp)-> exit name. Fixed arrival rate via a token queue."""
    stop=threading.Event(); q=queue.Queue(maxsize=nworkers*4)
    lat=[]; correct=[0]; total=[0]; exit_used=[]; lock=threading.Lock()
    cur_exit=[select_fn(temp())]
    def worker():
        while not stop.is_set():
            try: idx=q.get(timeout=0.5)
            except queue.Empty: continue
            ex=cur_exit[0]; t0=time.perf_counter()
            out=SESS[ex].run(None,{INAME:img(idx)})[0]
            dt=(time.perf_counter()-t0)*1000
            with lock:
                lat.append(dt); total[0]+=1
                if int(out.argmax(1)[0])==Y[idx]: correct[0]+=1
                exit_used.append(ex)
    ws=[threading.Thread(target=worker,daemon=True) for _ in range(nworkers)]
    for w in ws: w.start()
    # controller thread: update selected exit every 1s from temp
    temps=[]; viol=[0]
    def control():
        while not stop.is_set():
            T=temp(); temps.append(T)
            if T>cap: viol[0]+=1
            cur_exit[0]=select_fn(T)
            time.sleep(1.0)
    ct=threading.Thread(target=control,daemon=True); ct.start()
    # arrival generator at fixed rate
    t0=time.time(); i=0; interval=1.0/rate_hz
    while time.time()-t0<dur:
        try: q.put(i%10000, timeout=0.01)
        except queue.Full: pass
        i+=1; nxt=t0+i*interval; time.sleep(max(0,nxt-time.time()))
    stop.set(); time.sleep(1)
    lat=np.array(lat) if lat else np.array([0.0])
    from collections import Counter
    mix=Counter(exit_used)
    realacc=100*correct[0]/max(total[0],1)
    return dict(policy=name, n=total[0], real_acc=realacc,
                mean_temp=float(np.mean(temps)), max_temp=float(np.max(temps)),
                cap_viol_s=viol[0], p50=float(np.percentile(lat,50)),
                p99=float(np.percentile(lat,99)),
                exit_mix={k:round(100*v/len(exit_used),1) for k,v in mix.items()} if exit_used else {})

# --- REACTIVE controller: temp-feedback with hysteresis (no fragile util model) ---
# state = current exit index (0=exit1 shallow/cool, 2=final deep/hot).
# Each tick: too hot -> drop one level (cooler, less accurate);
#            comfortably cool -> raise one level (deeper, more accurate).
def make_controller(cap, band=3.0):
    idx=[2]  # start at final
    def sel(T):
        if T > cap and idx[0]>0: idx[0]-=1
        elif T < cap-band and idx[0]<2: idx[0]+=1
        return EXITS[idx[0]]
    return sel

def main():
    CAP=49.0                 # aggressive cap (final~51C violates, exit2~47C safe): binds
    RATE=1400                # req/s, just below final's ~1595/s saturation
    DUR=90
    print(f"Paper 3 controller experiment: rate={RATE}/s, cap={CAP}C, {DUR}s each")
    print(f"start temp {temp():.1f}C\n")
    print(f"{'policy':16s} {'n':>7s} {'acc':>7s} {'meanT':>6s} {'maxT':>6s} {'viol_s':>7s} {'p99':>7s} {'exit_mix'}")
    print("-"*88)
    results=[]
    policies=[("always-exit1",lambda T:"exit1"),
              ("always-exit2",lambda T:"exit2"),
              ("always-final",lambda T:"final"),
              ("adaptive-ctrl",make_controller(CAP))]
    for name,fn in policies:
        # cool to baseline first
        while temp()>50.5: time.sleep(2)
        r=run_policy(name,fn,RATE,CAP,DUR)
        results.append(r)
        print(f"{r['policy']:16s} {r['n']:7d} {r['real_acc']:6.2f}% {r['mean_temp']:5.1f}C "
              f"{r['max_temp']:5.1f}C {r['cap_viol_s']:6d}s {r['p99']:6.2f}ms {r['exit_mix']}")
        time.sleep(15)
    json.dump(results,open(Path.home()/"paper3_controller_results.json","w"),indent=2)
    print("\nController wins if: real_acc > always-exit1 AND cap_viol_s < always-final.")

if __name__=="__main__": main()
