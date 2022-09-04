#!/usr/bin/env bash

# signal patchset for the SUSY EWK 1Lbb analysis
curl -sL https://www.hepdata.net/record/resource/1408476?view=true | tar -xzv --one-top-level=1Lbb-likelihoods
cd 1Lbb-likelihoods

# verify patchset is valid
pyhf patchset verify BkgOnly.json patchset.json

# signal model: m1 = 900, m2 = 300
cat BkgOnly.json | \
pyhf cls --patch <(pyhf patchset extract --name C1N2_Wh_hbb_900_300 patchset.json) | \
jq .CLs_obs

# signal model: m1 = 900, m2 = 400
pyhf patchset extract --name C1N2_Wh_hbb_900_400 --output-file C1N2_Wh_hbb_900_400_patch.json patchset.json
pyhf cls --patch C1N2_Wh_hbb_900_400_patch.json BkgOnly.json | \
jq .CLs_obs
