import pickle
from hpo import all_archs, anchors

def read():
    f = open("./light/cora.bench", "rb")
    bench = pickle.load(f)
    f.close()

    fo = open("larges.bench", "w")

    max_p = 0
    valid_arch = 0
    best_arch = None
    archs = all_archs()
    for arch in archs:
        hash = arch.hash_arch()
        info = bench.get(hash, None)
        if info: 
            valid_arch += 1
            perf = info['perf']
            if perf > max_p:
                max_p = perf 
                best_arch = arch
        else:
            fo.write(str(hash) + '\n')
            print(arch.link)
            print(arch.ops)
            continue

    fo.close()
    print("Finish Writing.")

    print(max_p)
    print(best_arch.link)
    print(best_arch.ops)

    '''print("anchors")
    for i in anchors:
        h = i.hash_arch()
        info = bench.get(h)
        print(info['perf'])'''

    print("archs in bench: {}".format(len(bench)))
    print("valid in bench: {}".format(valid_arch))
    v = list(bench.values())
    v = [i['para'] for i in v]
    print(max(v))
    for key in bench:
        print(key)
        print(bench)
    return

read()
