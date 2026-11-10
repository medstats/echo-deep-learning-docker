"""Check length-to-shortening behavior without patient inference or model loading."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from worker import upstream_strain_functions

ROOT=Path(__file__).resolve().parents[1]
ns=upstream_strain_functions()
# Known periodic contour shortening: 170px ES /230px ED, before moving-average attenuation.
t=np.arange(240);lengths=(200+30*np.cos(2*np.pi*t/60)).tolist()
r=ns['calc_ratio'](lengths,'unused','synthetic.avi',save=False)
assert len(r)>=3 and np.isfinite(r).all(),r
signed=100*(float(np.mean(r))-1)
assert -27<signed<-24,(r,signed)
# Replaying the actual exported public-example length curve must reproduce returned ratios.
folder=ROOT/'strain_results/upstream_example/echonet_strain'
if (folder/'result.json').exists():
 saved=json.loads((folder/'result.json').read_text())
 if saved.get('status')=='ok':
  table=pd.read_csv(folder/'contours.csv')
  replay=ns['calc_ratio'](table['length'].tolist(),'unused','example.avi',save=False)
  np.testing.assert_allclose(replay,saved['beat_length_ratios'],rtol=1e-12,atol=1e-12)
  assert saved['upstream_positive_shortening_percent']==-saved['signed_longitudinal_contour_shortening_percent']
print('PASS: known periodic shortening, finite ratios, signed convention, replay from exported example lengths.')

# Verify the official Unity peak-based percentage is invariant to a constant spatial scale.
import sys
sys.path.insert(0,str(ROOT/'repos/unity-gls'))
from Scantensus.measure.lv import find_peaks_cm
percentages=[]
for scale in [1.0,0.01]:
    peaks=find_peaks_cm(np.asarray(lengths)*scale,min_name='ES',max_name='ED')
    ed=np.mean([x['value'] for x in peaks['ED']])
    es=np.mean([x['value'] for x in peaks['ES']])
    percentages.append(100*(es-ed)/ed)
np.testing.assert_allclose(percentages,[100*(170-230)/230]*2,rtol=1e-12,atol=1e-12)
print('PASS: Unity official peak selection, negative shortening sign and constant-scale invariance.')
