from pathlib import Path
home = Path(__file__).resolve().parent.parent
data_path = home / "data" / "raw"
print(f"dir stem: {home.stem}")
print(f"anchor: {home.anchor}")
print(f"name: {Path(__file__).name}")
print(f"file stem: {Path(__file__).stem}")
print(f"suffix: {Path(__file__).suffix}")

print(Path(__file__).resolve().parent)
print(home)
print(data_path)
files = list(data_path.glob("*"))
f = files[1]

from pandas import read_csv
# data = read_csv
print(f"file: {f}")
# data = read_csv(f)
# print(data.head())
print(f.stat().st_size)

print(f.stem)
if "btl" in f.name:
    print("btl")
else:
    print("not btl")