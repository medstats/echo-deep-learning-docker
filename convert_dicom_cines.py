import os, sys, re, time, pydicom, cv2, json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def sanitize(name):
    clean = re.sub(r'[\^]+', '_', name)
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]+', '_', clean)
    return clean.strip('_')

def convert_single_dicom(args):
    dicom_path, out_dir, index = args
    try:
        ds = pydicom.dcmread(dicom_path)
        frames = int(ds.get('NumberOfFrames', 1))
        if frames <= 1:
            return None # static image or non-cine
        
        # Determine FPS
        cine_rate = ds.get('CineRate')
        frame_time = ds.get('FrameTime')
        if cine_rate and float(cine_rate) > 0:
            fps = float(cine_rate)
        elif frame_time and float(frame_time) > 0:
            fps = 1000.0 / float(frame_time)
        else:
            fps = 30.0
        fps = max(10.0, min(fps, 100.0))
        
        arr = ds.pixel_array
        if arr.ndim == 3: # grayscale (frames, H, W) -> expand to 3ch
            arr = np.repeat(arr[:, :, :, None], 3, axis=-1)
        elif arr.ndim != 4:
            return None
        
        num_f, h, w, ch = arr.shape
        out_filename = f"cine_{index:02d}.mp4"
        out_file = out_dir / out_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
        for frame in arr:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
        vw.release()
        
        return {
            'file': str(out_file),
            'filename': out_filename,
            'source_dicom': dicom_path.name,
            'frames': num_f,
            'fps': fps,
            'width': w,
            'height': h,
            'series_desc': str(ds.get('SeriesDescription', '')),
            'instance_num': int(ds.get('InstanceNumber', index))
        }
    except Exception as e:
        return {'file': None, 'error': f"{dicom_path.name}: {str(e)}"}

def main():
    start_time = time.time()
    dicom_root = Path('/workspace/echo_inputs/echos/DICOM')
    patients_root = Path('/workspace/patients')
    patients_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover all files and group by patient
    patient_dicoms = {}
    for f in sorted(dicom_root.glob('*')):
        if not f.is_file() or f.name.startswith('SR_'):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            frames = int(ds.get('NumberOfFrames', 1))
            if frames > 1:
                pname = sanitize(str(ds.get('PatientName', 'UNKNOWN')))
                pid = sanitize(str(ds.get('PatientID', 'UNKNOWN')))
                key = f"{pname}__{pid}"
                if key not in patient_dicoms:
                    patient_dicoms[key] = []
                patient_dicoms[key].append(f)
        except Exception as e:
            pass
            
    print(f"Found {len(patient_dicoms)} patients with multi-frame cines.")
    
    conversion_jobs = []
    cohort_manifest = {}
    for p_key, files in patient_dicoms.items():
        p_dir = patients_root / p_key
        p_dir.mkdir(parents=True, exist_ok=True)
        cohort_manifest[p_key] = {'dir': str(p_dir), 'cines': []}
        for idx, f in enumerate(files, 1):
            conversion_jobs.append((f, p_dir, idx))
            
    print(f"Total cine conversion jobs: {len(conversion_jobs)}. Running with multiprocessing...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        for res in pool.map(convert_single_dicom, conversion_jobs):
            if res and res.get('file'):
                p_key = Path(res['file']).parent.name
                cohort_manifest[p_key]['cines'].append(res)
                success_count += 1
                
    elapsed = time.time() - start_time
    print(f"\nSuccessfully converted {success_count} / {len(conversion_jobs)} cines in {elapsed:.1f}s!")
    
    manifest_path = Path('/workspace/cohort_manifest.json')
    manifest_path.write_text(json.dumps(cohort_manifest, indent=2))
    print(f"Saved cohort manifest to {manifest_path}")

if __name__ == '__main__':
    main()
