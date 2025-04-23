import os


maindir = r"c:\Data\PhD\palimpsest\Victor_data\Paris_Coislin"
folioname = r"Par_coislin_393_054r"

banddict={}
for fname in os.listdir(os.path.join(maindir,folioname)):
    if "tif" in fname:
        band = fname.split("+")[1][:-4]
        band_idx = int(band.split("_")[1])
        banddict[band_idx]=band

bands = []

for idx in sorted(banddict.keys()):
    print(idx)
    bands.append(banddict[idx]+"\n")

with open(os.path.join(maindir,"band_list.txt"),"w") as f:
    f.writelines(bands)
