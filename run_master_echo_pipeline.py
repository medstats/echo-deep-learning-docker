import os, sys, re, time, json, subprocess
from pathlib import Path

def main():
    start_total = time.time()
    workspace = Path('/workspace')
    patients_dir = workspace / 'patients'
    output_dir = workspace / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_dirs = sorted([p for p in patients_dir.iterdir() if p.is_dir()])
    print(f"======================================================================")
    print(f"MASTER ECHO MULTI-MODEL INFERENCE PIPELINE")
    print(f"Cohort Size: {len(patient_dirs)} patients")
    print(f"Output Path: {output_dir}")
    print(f"======================================================================\n")
    
    # -------------------------------------------------------------------------
    # STAGE 1: EchoPrime (Full Study Video Encoding, View Classification, 21 Metrics, Report)
    # -------------------------------------------------------------------------
    print(f"\n>>> STAGE 1: Running EchoPrime Suite (21 Metrics, View Classifier, Reports)...")
    ep_out = output_dir / 'echoprime'
    cmd_ep = [
        sys.executable, str(workspace / 'echoprime_cloud.py'),
        '--input-root', str(patients_dir),
        '--output-dir', str(ep_out),
        '--cache-dir', str(workspace / 'cache/echoprime'),
        '--spatial-mode', 'adaptive'
    ]
    subprocess.run(cmd_ep, check=False)
    
    # Parse detected views per patient from EchoPrime
    patient_views = {}
    for p_dir in patient_dirs:
        p_name = p_dir.name
        ep_json = ep_out / p_name / 'echoprime.json'
        if ep_json.exists():
            try:
                data = json.loads(ep_json.read_text())
                views = {}
                for vid in data.get('input_videos', []):
                    fname = vid.get('file')
                    pred_v = vid.get('predicted_view')
                    if fname and pred_v:
                        views[fname] = pred_v
                patient_views[p_name] = views
            except Exception as e:
                print(f"Error parsing EchoPrime JSON for {p_name}: {e}")
                
    # -------------------------------------------------------------------------
    # STAGE 2: Rename/Link A4C cines for EchoNet-Dynamic and Strain
    # -------------------------------------------------------------------------
    print(f"\n>>> STAGE 2: Tagging Best A4C Cines per Patient based on View Classifier...")
    patient_a4c = {}
    for p_dir in patient_dirs:
        p_name = p_dir.name
        views = patient_views.get(p_name, {})
        # Find videos labeled as A4C (or apical 4 chamber)
        a4c_candidates = []
        for vid_name, vlabel in views.items():
            if 'A4C' in vlabel.upper() or 'APICAL 4' in vlabel.upper() or '4CH' in vlabel.upper():
                a4c_candidates.append(vid_name)
        
        # If no strict A4C, fallback to first cine
        all_cines = sorted(list(p_dir.glob('*.mp4')))
        best_a4c = None
        if a4c_candidates:
            best_a4c = p_dir / a4c_candidates[0]
        elif all_cines:
            best_a4c = all_cines[0]
            
        if best_a4c and best_a4c.exists():
            # Create a symbolic link A4C.mp4 in the patient directory for standard runners
            a4c_symlink = p_dir / 'A4C.mp4'
            if not a4c_symlink.exists():
                try:
                    os.symlink(best_a4c.name, str(a4c_symlink))
                except Exception:
                    pass
            patient_a4c[p_name] = str(best_a4c)
            print(f"  {p_name:<35} -> Selected A4C Cine: {best_a4c.name}")

    # -------------------------------------------------------------------------
    # STAGE 3: PanEcho (40 Diagnostic Tasks across all cines)
    # -------------------------------------------------------------------------
    print(f"\n>>> STAGE 3: Running PanEcho (40 Diagnostic & Hemodynamic Tasks)...")
    for p_dir in patient_dirs:
        p_name = p_dir.name
        out_p = output_dir / p_name / 'panecho.json'
        out_p.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Processing PanEcho for {p_name}...")
        cmd_pan = [
            'sh', str(workspace / 'runtime/python'),
            str(workspace / 'infer_worker.py'),
            '--model', 'panecho',
            '--patient', str(p_dir),
            '--output', str(out_p),
            '--clips', '4',
            '--threads', '8'
        ]
        subprocess.run(cmd_pan, check=False)

    # -------------------------------------------------------------------------
    # STAGE 4: EchoNet-Dynamic (Biplane LVEF & Simpson's Chamber Tracking)
    # -------------------------------------------------------------------------
    print(f"\n>>> STAGE 4: Running EchoNet-Dynamic (A4C LVEF Estimation)...")
    for p_dir in patient_dirs:
        p_name = p_dir.name
        out_e = output_dir / p_name / 'echonet.json'
        print(f"  Processing EchoNet for {p_name}...")
        cmd_echo = [
            'sh', str(workspace / 'runtime/python'),
            str(workspace / 'infer_worker.py'),
            '--model', 'echonet',
            '--patient', str(p_dir),
            '--output', str(out_e),
            '--clips', '4',
            '--threads', '8'
        ]
        subprocess.run(cmd_echo, check=False)

    # -------------------------------------------------------------------------
    # STAGE 5: EchoCLIP (Zero-Shot EF & Diagnostic Text Alignment)
    # -------------------------------------------------------------------------
    print(f"\n>>> STAGE 5: Running EchoCLIP (Zero-Shot EF)...")
    for p_dir in patient_dirs:
        p_name = p_dir.name
        out_c = output_dir / p_name / 'echoclip.json'
        print(f"  Processing EchoCLIP for {p_name}...")
        cmd_clip = [
            'sh', str(workspace / 'runtime/python'),
            str(workspace / 'infer_worker.py'),
            '--model', 'echoclip',
            '--patient', str(p_dir),
            '--output', str(out_c),
            '--clips', '4',
            '--threads', '8'
        ]
        subprocess.run(cmd_clip, check=False)

    # -------------------------------------------------------------------------
    # STAGE 6: Strain Analysis (Unity-GLS and EchoNet-Strain on A4C)
    # -------------------------------------------------------------------------
    print(f"\n>>> STAGE 6: Running Myocardial Strain (Unity-GLS & EchoNet-Strain)...")
    for p_dir in patient_dirs:
        p_name = p_dir.name
        a4c_path = patient_a4c.get(p_name)
        if a4c_path and os.path.exists(a4c_path):
            strain_out = output_dir / p_name / 'strain'
            strain_out.mkdir(parents=True, exist_ok=True)
            print(f"  Processing Strain for {p_name} on {Path(a4c_path).name}...")
            for model_name in ['unity_gls', 'echonet_strain']:
                m_out = strain_out / model_name
                m_out.mkdir(parents=True, exist_ok=True)
                cmd_strain = [
                    'sh', str(workspace / 'runtime/strain-python'),
                    str(workspace / 'strain/worker.py'),
                    '--model', model_name,
                    '--input', str(a4c_path),
                    '--view', 'A4C',
                    '--output', str(m_out),
                    '--threads', '8'
                ]
                subprocess.run(cmd_strain, check=False)

    elapsed = time.time() - start_total
    print(f"\n======================================================================")
    print(f"ALL INFERENCE COMPLETE IN {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"======================================================================\n")

if __name__ == '__main__':
    main()
