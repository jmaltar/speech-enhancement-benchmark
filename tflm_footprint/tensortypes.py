import sys, collections
from tflite.Model import Model
from tflite.TensorType import TensorType
tn = {v: k for k, v in vars(TensorType).items() if isinstance(v, int)}
for path in sys.argv[1:]:
    m = Model.GetRootAsModel(bytearray(open(path,'rb').read()), 0)
    sg = m.Subgraphs(0)
    print(f"\n=== {path} ===")
    for lbl, idxs in (("inputs", [sg.Inputs(i) for i in range(sg.InputsLength())]),
                      ("outputs",[sg.Outputs(i) for i in range(sg.OutputsLength())])):
        for i in idxs:
            t = sg.Tensors(i)
            print(f"  {lbl:<8} {t.Name().decode():<40} {tn[t.Type()]:<8} "
                  f"shape {[t.Shape(j) for j in range(t.ShapeLength())]}")
    c = collections.Counter()
    wb = collections.Counter()
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        n = m.Buffers(t.Buffer()).DataLength()
        if n:
            c[tn[t.Type()]] += 1
            wb[tn[t.Type()]] += n
    print("  constant tensors by type:", {k: f"{c[k]} tensors / {wb[k]:,} B" for k in sorted(c)})
