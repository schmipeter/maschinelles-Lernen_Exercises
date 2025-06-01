import scipy.io as sio, pathlib

# ► richtiger Pfad relativ zum Projekt-Root
mat = pathlib.Path("data/dogData_w.mat")      # ← Pfad angepasst!
data = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)

print("Variablen in", mat.name + ":")
for k, v in data.items():
    if k.startswith("__"):
        continue
    print(f" - {k:12s}  shape={getattr(v, 'shape', None)}  dtype={getattr(v, 'dtype', None)}")