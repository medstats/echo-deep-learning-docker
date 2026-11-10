"""Build the installation report from actual worker outputs and pinned assets."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'weights/manifest.json').read_text())
packages=json.loads((ROOT/'runtime/package_versions.json').read_text())
rows=[];results={}
for name in ['echonet','panecho','echoclip']:
 p=ROOT/'results/ECHO_01'/f'{name}.json'
 d=json.loads(p.read_text()) if p.exists() else {'status':'pending'}
 results[name]=d
 wall=d.get('monitor',{}).get('wall_seconds_including_imports')
 rows.append('| '+ ' | '.join([name,d['status'],f"{manifest[name]['bytes']/1e6:.1f} MB",str(d.get('parameter_count','—')),f"{d['inference_seconds']:.2f}" if 'inference_seconds' in d else '—',f'{wall:.2f}' if wall is not None else '—',f"{d['peak_rss_mib']:.1f}" if 'peak_rss_mib' in d else '—'])+' |')
lines=['# Installation and inference report','', '25 October 2026 — Android aarch64 / SM8750, CPU FP32. This report is generated from actual checkpoint inference outputs.','',
'## Verified execution','',
'| Model | Patient run | Official checkpoint | Parameters | Patient inference (s) | Process wall time (s) | Peak RSS (MiB) |',
'|---|---|---:|---:|---:|---:|---:|']+rows
lines+=['','Patient inference excludes loading and synthetic checks; process wall time includes imports, loading, synthetic checks and patient inference. Peak RSS includes loading and synthetic checks and is measured by getrusage; the launcher independently samples process RSS and system available RAM. Threads=3; one model process at a time; no CUDA, training or quantisation.','',
'## Actual outputs for ECHO_01','']
for n,d in results.items():
 if d['status'] not in ['ok','partial']:
  lines.append(f'- {n}: {d["status"]}. Errors: {d.get("errors",[])}');continue
 pr=d.get('predictions',{})
 if n=='panecho':
  lines.append(f'- PanEcho: {len(pr)} task outputs across {len(d["videos"])} videos. EF estimate {pr.get("EF",{}).get("value")}%; LV wall-motion abnormality probability {pr.get("LVWallMotionAbnormalities",{}).get("probability")}.')
 else:
  lines.append(f'- {n}: EF estimate {pr.get("ef_percent")}% across {len(d["videos"])} video(s).'+(f' Pacemaker cosine similarity {pr.get("pacemaker_similarity_score")} (not a probability).' if n=='echoclip' else ''))
 lines.append(f'  Synthetic forward passed: {d.get("synthetic_test",{}).get("passed")}. Full details: [results/ECHO_01/{n}.json](results/ECHO_01/{n}.json).')
lines+=['','No clinical reference labels were supplied. Successful model inference is not a validation of accuracy on this patient. Models use different tasks, normalization and aggregation; numerical agreement or disagreement does not establish the true EF.','',
'## Installation and isolation','',
'- Official repositories are cloned under repos/dynamic, repos/PanEcho and repos/echo_CLIP. Exact commit IDs are in the output JSON. Original model source is retained; adaptations live in infer_worker.py.',
'- Reused existing glibc Python3.12 CPU torch/torchvision/OpenCV from ~/ecg-ai/venv, through sh runtime/python. Existing environments and project files were not upgraded or modified.',
'- Added only open_clip_torch3.3.0, timm1.0.30 and ftfy6.3.1 in /data/data/com.termux/files/home/.local/share/echo-inference/deps. Existing prerequisites are reused. Exact runtime package versions: runtime/package_versions.json.',
'- All three checkpoints came from official publisher URLs, without training data downloads. SHA256 and byte counts: weights/manifest.json. EchoCLIP digest matches its author-published Hugging Face LFS SHA256; source revision and digest are saved in weights/echoclip/source_metadata.json.',
'- PanEcho generic ImageNet initialization download is suppressed because every learned backbone weight is replaced by the official PanEcho checkpoint. The only allowed missing key is its fixed positional encoding, matching official hubconf.py behavior.',
'- Inference uses local weights with Hugging Face offline mode. No patient videos or images were sent to an external API.','',
'## Preprocessing and practical limits','',
'### EchoNet-Dynamic',
'Filename-labelled A4C only; official square/border crop, scan-sector mask, cubic112 resize, grayscale repeated across3channels;32frames with stride2, four evenly spaced clips. The original notebook first-black-row heuristic was unsafe on already-cleaned sector-only MP4s: it initially selected the lower border and erased anatomy. A visual preprocessing check caught this before model inference; header trimming now applies only if the first black row is in the top15%. The resulting frame was visually inspected.',
'',
'Input-cohort channel mean/std follow the official external-test notebook approach. Here all supplied A4C frames are used deterministically. This is an adaptation, not verified original training-cohort normalization. The small single-patient normalization cohort limits calibration. Output is clip-averaged EF; LV segmentation and beat-by-beat analysis were not installed.',
'',
'### PanEcho',
'Incremental MP4 decode; OpenCV BGR convention retained as in official loader; resize to256x256 using INTER_AREA (same resizer as released external dataset loader), center crop224, FP32/ImageNet normalization. Normalization/crop checked against Torchvision v2 official evaluation transform: maximum difference7.2e-7. Four deterministic16-frame clips per video, mean activated clip probabilities per video then mean across videos. Original direct reshape in official forward is preserved. GLS physical sign follows tasks.md.',
'',
'Current inputs are68fps. Retaining original frame rate means16frames span only~0.22s; this may affect transfer to these exports. Upstream acquisition/sector preprocessing parity is not fully established. No undocumented temporal resampling or architecture correction was introduced.',
'',
'Wall-motion output is a study-level probability, not a segment map or a validated exclusion of RWMA. The study has A4C/A3C/PLAX/PSAX but no filename-labelled A2C. Doppler-dependent quantities and valve-severity outputs are model estimates from limited2Dviews, not acquired haemodynamic measurements.',
'',
'### EchoCLIP',
'Standard released EchoCLIP, CPU FP32. Official crop_and_scale to224 with10%zoom; checkpoint-specific OpenCLIP normalization. Original OpenCV BGR-to-PIL convention is retained; these input loops have identical grayscale channels. First40frames sampled at stride2, image batch1; text prompts encoded one at a time. Official EF prompt-ranking/average-rank/top20%median function is used. Study-level pooling of frames across views is an explicit exploratory extension of the single-video example. Pacemaker output remains an uncalibrated similarity score; no thresholded diagnosis or RWMA output is fabricated.',
'',
'## Errors and recovery','',
'- No model is considered working merely because its repository cloned. Consult the execution table and output status above.',
'- A first EchoNet launcher attempt stopped before loading on Git shared-storage ownership protection. Fixed using a process-scoped safe.directory for the exact repository; global Git config was not changed. The rerun passed.',
'- Initial single-stream official downloads were extremely slow; switched to validated resumable byte-range transfers. Completed byte counts and hashes are recorded.',
'- Any remaining model failures are recorded in the output JSON and corresponding logs/.','',
'## Run commands','',
'From /storage/emulated/0/echo-inference:','',
'```sh',
'python run_inference.py --patient patients/ECHO_01',
'python run_inference.py',
'python run_inference.py --patient patients/ECHO_01 --models panecho',
'python run_inference.py --patient patients/ECHO_01 --smoke-only',
'```','',
'Use --clips N and --threads N to change the EchoNet/PanEcho clip count and CPU threads. EchoCLIP retains its official example frame selection. Results: results/<patient>/{echonet,panecho,echoclip}.json. Synthetic-only outputs: smoke_results/. Memory monitor stops a model above3584MiB RSS or sustained system available RAM below384MiB and continues other models.','',
'## Additional requested model research','',
'See [model_research.md](model_research.md). EchoFM evaluates view classification, LV/LA segmentation, aortic-stenosis screening and color-Doppler AR severity using task-specific heads; public quickstart yields embeddings. Echo-Vision-FM evaluates EF, EDV/ESV, heart-function and aortic-stenosis classification; its released checkpoint is described as a pretrained backbone. Matching turnkey clinical task heads for those releases were not verified. Neither was installed. EchoPrime remains deferred.','',
'## Checkpoint provenance','']
for n,m in manifest.items():lines += [f'- {n}: {m["url"]}',f'  SHA256: `{m["sha256"]}`; {m["bytes"]} bytes.']
(ROOT/'installation_report.md').write_text('\n'.join(lines)+'\n')
print('installation_report.md updated')
