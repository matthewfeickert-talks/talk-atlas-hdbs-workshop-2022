import json
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhf.contrib.viz.brazil import plot_results

pyhf.set_backend("jax")  # Optional for speed
spec = json.load(open("1Lbb-pallet/BkgOnly.json"))
patchset = pyhf.PatchSet(json.load(open("1Lbb-pallet/patchset.json")))
workspace = pyhf.Workspace(spec)
model = workspace.model(patches=[patchset["C1N2_Wh_hbb_900_250"]])
test_pois = np.linspace(0, 5, 41)  # POI step of 0.125
data = workspace.data(model)
obs_limit, exp_limits, (test_pois, results) = pyhf.infer.intervals.upperlimit(
    data, model, test_pois, return_results=True
)
print(f"Observed limit: {obs_limit}")
# Observed limit: 2.547958147632675
print(f"Expected limits: {[limit.tolist() for limit in exp_limits]}")
# Expected limits: [0.7065311975182036, 1.0136453820160332,
# 1.5766626372587724, 2.558234487679955, 4.105381941514062]
fig, ax = plt.subplots()
artists = plot_results(test_pois, results, ax=ax)
fig.savefig("upper_limit.pdf")
