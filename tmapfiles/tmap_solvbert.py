import pandas as pd
import tmap as tm
from mhfp.encoder import MHFPEncoder
from faerun import Faerun
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
import numpy as np
import torch
from tqdm import tqdm

df = pd.read_csv("../data/Exp.csv")

list_main_smi = df['ssid'].tolist()

c_list = df['dgsolv'].tolist()
# compute reaction fingerprint
model, tokenizer = get_default_model_and_tokenizer('exp')
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
exp_fp = generate_fingerprints(list_main_smi, rxnfp_generator, batch_size=8)
np.savez_compressed('exp_fp', fps=exp_fp)
exp_fp = np.load('exp_fp.npz')['fps']

labels=[]
for index, i in df.iterrows():
    label = (
        i['ssid'] +
        "__<h1>" + str(i['dgsolv']) + "kcal/mol" + "</h1>"
    )
    labels.append(label)


# dims = 512
dims = 256

enc = tm.Minhash(dims)
lf = tm.LSHForest(dims, 128)

# fingerprints = [tm.VectorUint(enc.encode(s)) for s in list_main_smi]　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
fingerprints = [enc.from_weight_array(fp.tolist(), method="ICWS") for fp in tqdm(exp_fp)]


lf.batch_add(fingerprints)
lf.index()

cfg = tm.LayoutConfiguration()
cfg.k = 50
cfg.kc = 50
cfg.sl_scaling_min = 1.0
cfg.sl_scaling_max = 1.0
cfg.sl_repeats = 1
cfg.sl_extra_scaling_steps = 2
cfg.placer = tm.Placer.Barycenter
cfg.merger = tm.Merger.LocalBiconnected
cfg.merger_factor = 2.0
cfg.merger_adjustment = 0
cfg.fme_iterations = 1000
cfg.sl_scaling_type = tm.ScalingType.RelativeToDesiredLength
cfg.node_size = 1 / 37
cfg.mmm_repeats = 1

x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=cfg)

faerun = Faerun(coords=False, view='front', )
faerun.add_scatter(
    "dGsolv",
    {"x": x,
     "y": y,
     "c": [c_list],
     "labels": labels},
    point_scale=3,
    colormap='rainbow',
    has_legend=True,
    categorical=False,
    series_title="dGsolv",
    shader='smoothCircle'

)
faerun.add_tree("dGsolv_tree", {"from": s, "to": t}, point_helper="dGsolv")

faerun.plot('pretrain_finetune', template="smiles", notebook_height=750)
