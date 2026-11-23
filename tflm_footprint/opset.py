import sys, collections
from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator as BO

names = {v: k for k, v in vars(BO).items() if isinstance(v, int)}

for path in sys.argv[1:]:
    buf = bytearray(open(path, 'rb').read())
    m = Model.GetRootAsModel(buf, 0)
    codes = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        bc = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        codes.append(names.get(bc, f"UNKNOWN({bc})"))
    used = collections.Counter()
    tensor_bytes = 0
    for s in range(m.SubgraphsLength()):
        sg = m.Subgraphs(s)
        for o in range(sg.OperatorsLength()):
            used[codes[sg.Operators(o).OpcodeIndex()]] += 1
    # weight bytes = sum of buffers actually populated
    for b in range(m.BuffersLength()):
        buf_i = m.Buffers(b)
        if buf_i.DataLength():
            tensor_bytes += buf_i.DataLength()
    print(f"\n=== {path} ===")
    print(f"  file {len(buf):,} B, populated buffers {tensor_bytes:,} B "
          f"({100*tensor_bytes/len(buf):.1f}% of file)")
    print(f"  {sum(used.values())} operators, {len(used)} distinct:")
    for k, v in sorted(used.items()):
        print(f"    {v:>3}x  {k}")
