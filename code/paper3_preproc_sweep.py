import numpy as np, tarfile, pickle, onnxruntime as ort
from pathlib import Path
DATA=Path.home()/"Desktop/researchpaper3/data"
with tarfile.open(DATA/"cifar-100-python.tar.gz","r:gz") as t:
    d=pickle.loads(t.extractfile("cifar-100-python/test").read(),encoding="latin1")
raw=np.asarray(d["data"],dtype=np.uint8); y=np.array(d["fine_labels"],dtype=np.int64)
s=ort.InferenceSession(str(Path.home()/"tier1-experiments/models/mnv3_c100_final_fp32.onnx"),
                       providers=["CPUExecutionProvider"])
iname=s.get_inputs()[0].name
N=2000
def acc(fn):
    c=0
    for k in range(0,N,200):
        x=fn(raw[k:k+200]); lg=s.run(None,{iname:x})[0]; c+=int((lg.argmax(1)==y[k:k+200]).sum())
    return 100*c/N
base=lambda r: r.reshape(-1,3,32,32).astype(np.float32)
def norm(m,sd):
    m=np.array(m,np.float32)[:,None,None]; sd=np.array(sd,np.float32)[:,None,None]
    return lambda r:(base(r)/255.0-m)/sd
variants={
 "cifar100_norm": norm([0.5071,0.4865,0.4409],[0.2673,0.2564,0.2762]),
 "0_1_scale": lambda r: base(r)/255.0,
 "imagenet_norm": norm([0.485,0.456,0.406],[0.229,0.224,0.225]),
 "neg1_1": lambda r: base(r)/127.5-1.0,
 "raw_uint8": lambda r: base(r),
 "cifar10_norm": norm([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]),
}
print("preprocessing sweep, mnv3_c100_final_fp32 (2000 imgs, random=1%):")
for name,fn in variants.items():
    try: print(f"  {name:16s} acc={acc(fn):.2f}%")
    except Exception as e: print(f"  {name:16s} ERR {e}")
