"""Compare the training-time fake-quant scales of a --wq-only run with the scales the 16x8
per-tensor-dense converter wrote (2026-10-22). Usage: .venv/bin/python training/scale_check.py training/runs/wqat3
Encoder/decoder: per-output-channel absmax/127 over the authors-layout kernel; dense2: per-tensor over the
FULL 128-row kernel (what training saw) vs the live rows the shrunk export carries."""
import numpy as np, h5py, sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
run=sys.argv[1]
f=h5py.File(f'{run}/best.authors.weights.h5','r')
enc=f['/layers/conv1d/vars/0'][0]        # (512,256) in,out
dec=f['/layers/conv1d_1/vars/0'][0]      # (256,512)
d2 =f['/layers/dense_1/vars/0'][()]      # (128,256) full-size mask dense
def details(p):
    it=tf.lite.Interpreter(model_path=p); it.allocate_tensors()
    out={}
    for d in it.get_tensor_details():
        try: out[d['index']]=(d, it.get_tensor(d['index']))
        except Exception: pass
    return out
q=details(f'{run}/shrunk/model_i16pt_2.tflite'); fl=details(f'{run}/shrunk/model_f32_2.tflite')
def find(dets, dtype, shape):
    r=[(d,t) for d,t in dets.values() if t.dtype==dtype and tuple(t.shape)==shape]
    assert len(r)==1,(dtype,shape,[(d['name'],t.shape) for d,t in r])
    return r[0]
# flatbuffer tensors
qenc,tenc=find(q,np.int8,(256,1,1,512)); qdec,tdec=find(q,np.int8,(512,1,1,256))
fcs=[(d,t) for d,t in q.values() if t.dtype==np.int8 and t.ndim==2 and t.shape[1]<=128]
print("int8 FC weight tensors in i16pt stage 2:",[(d['name'][:40],t.shape) for d,t in fcs])
fd2=[(d,t) for d,t in fl.values() if t.dtype==np.float32 and t.ndim==2 and t.shape[0]==256 and t.shape[1]<=128]
print("f32 FC tensors (256,<=128):",[(d['name'][:40],t.shape) for d,t in fd2])
# live rows of dense2: match shrunk f32 kernel (256,k) columns to full-size rows
fw=fd2[0][1]              # (256,k) tflite layout = out x in
k=fw.shape[1]
live=[]
for j in range(k):
    col=fw[:,j]
    m=np.where(np.all(np.isclose(d2,col[None,:],atol=0,rtol=0),axis=1))[0]
    live.append(int(m[0]) if len(m) else -1)
print("live dense2 rows:",live)
qd2,td2=[x for x in fcs if x[1].shape==(256,k)][0]
def cmp(name, w_train_axis, w, qd, qt, axis):
    """w: float kernel in training layout; axis: training per-channel axis or None"""
    qmax=127.0
    if axis is None:
        s_tr=np.array([max(np.abs(w).max(),1e-8)/qmax])
    else:
        red=tuple(i for i in range(w.ndim) if i!=axis)
        s_tr=np.maximum(np.abs(w).max(axis=red),1e-8)/qmax
    s_fb=np.array(qd['quantization_parameters']['scales']); zp=np.array(qd['quantization_parameters']['zero_points'])
    print(f"\n== {name}: training scales {s_tr.shape}, flatbuffer scales {s_fb.shape}, zero points unique {np.unique(zp)}")
    if s_tr.shape==s_fb.shape:
        r=s_fb/s_tr
        print(f"   ratio fb/train: min {r.min():.6f} max {r.max():.6f} mean {r.mean():.6f}; identical to 1e-6 rel: {np.allclose(s_fb,s_tr,rtol=1e-6)}")
    else:
        print(f"   train {s_tr[:3]} fb {s_fb[:3]}")
    return s_tr,s_fb
# encoder: training kernel (512 in,256 out), per-channel axis 1 (out). flatbuffer (256 out,1,1,512 in), per-channel axis 0
s_tr,s_fb=cmp("encoder basis",1,enc,qenc,tenc,1)
qt=np.round(enc/s_tr[None,:]).clip(-128,127).astype(np.int8)       # training-snapped ints (in,out)
fb=tenc[:,0,0,:].T                                                   # (in,out)
print("   int8 values differ:", int((qt!=fb).sum()), "of", qt.size, " max |diff|", int(np.abs(qt.astype(int)-fb.astype(int)).max()))
s_tr,s_fb=cmp("decoder basis",1,dec,qdec,tdec,1)
qt=np.round(dec/s_tr[None,:]).clip(-128,127).astype(np.int8); fb=tdec[:,0,0,:].T
print("   int8 values differ:", int((qt!=fb).sum()), "of", qt.size, " max |diff|", int(np.abs(qt.astype(int)-fb.astype(int)).max()))
# dense2: training per-tensor over FULL (128,256); flatbuffer over the live (k,256) rows
d2live=d2[live,:]
print(f"\ndense2 absmax: full kernel {np.abs(d2).max():.5f}  live rows only {np.abs(d2live).max():.5f}  dead rows only {np.abs(np.delete(d2,live,axis=0)).max():.5f}")
s_tr,s_fb=cmp("mask dense (dense2), training = full-size per-tensor",None,d2,qd2,td2,None)
s_live=np.abs(d2live).max()/127
print(f"   live-rows-only scale {s_live:.7g} vs flatbuffer {s_fb[0]:.7g}  ratio {s_fb[0]/s_live:.6f}")
qt=np.round(d2live/s_tr[0]).clip(-128,127).astype(np.int8); fb=td2.T  # (k,256)
print("   int8 values (training grid) vs flatbuffer differ:", int((qt!=fb).sum()), "of", qt.size, " max |diff|", int(np.abs(qt.astype(int)-fb.astype(int)).max()))
print("   training grid step / flatbuffer step =", s_tr[0]/s_fb[0])
