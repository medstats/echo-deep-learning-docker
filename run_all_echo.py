#!/usr/bin/env python3
"""
run_all_echo.py - Unified Master Inference CLI for all 6 Echo Models
Supports: PanEcho, EchoNet-Dynamic, EchoCLIP, Unity-GLS, EchoNet-Strain, EchoPrime
"""
import os, sys, argparse, subprocess, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Unified Echo Deep Learning Inference Runner")
    parser.add_argument('--patient', type=str, required=True, help="Path to patient study directory containing MP4 cines (or single MP4 for strain)")
    parser.add_argument('--output', type=str, default="results", help="Output directory for predictions and reports")
    parser.add_argument('--models', nargs='+', default=['all'], 
                        choices=['all', 'panecho', 'echonet', 'echoclip', 'unity_gls', 'echonet_strain', 'echoprime'],
                        help="Models to execute (default: all)")
    parser.add_argument('--a4c', type=str, default=None, help="Explicit A4C cine path (required for EchoNet-Dynamic / Strain if not auto-detected)")
    args = parser.parse_args()

    patient_dir = Path(args.patient)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    selected = set(args.models)
    if 'all' in selected:
        selected = {'panecho', 'echonet', 'echoclip', 'unity_gls', 'echonet_strain', 'echoprime'}

    print("="*70)
    print(f"ECHO DEEP LEARNING INFERENCE RUNNER")
    print(f"Patient Source : {patient_dir}")
    print(f"Output Target  : {out_dir}")
    print(f"Active Models  : {', '.join(sorted(selected))}")
    print("="*70)

    # 1. Standard Suite: PanEcho, EchoNet, EchoCLIP
    std_models = [m for m in ['panecho', 'echonet', 'echoclip'] if m in selected]
    if std_models:
        print(f"\n[1/3] Running Standard Suite: {', '.join(std_models)}...")
        cmd = [sys.executable, 'run_inference.py', '--patient', str(patient_dir), '--models', *std_models, '--output', str(out_dir)]
        subprocess.run(cmd, check=False)

    # 2. Strain Suite: Unity-GLS, EchoNet-Strain
    strain_models = [m for m in ['unity_gls', 'echonet_strain'] if m in selected]
    if strain_models:
        print(f"\n[2/3] Running Strain Suite: {', '.join(strain_models)}...")
        # Find A4C file
        a4c_file = args.a4c
        if not a4c_file:
            # Search inside patient directory
            cines = list(patient_dir.glob("*.mp4")) if patient_dir.is_dir() else [patient_dir]
            for c in cines:
                if 'A4C' in c.name.upper() or 'APICAL4' in c.name.upper() or '4CH' in c.name.upper():
                    a4c_file = str(c)
                    break
            if not a4c_file and cines:
                a4c_file = str(cines[0]) # fallback to first cine
        
        if a4c_file and os.path.exists(a4c_file):
            print(f"  Target Cine for Strain: {a4c_file}")
            for sm in strain_models:
                cmd = [sys.executable, 'run_strain.py', '--models', sm, '--input', a4c_file, '--view', 'A4C', '--output', str(out_dir / 'strain')]
                subprocess.run(cmd, check=False)
        else:
            print("  [Warning] No A4C cine found for strain analysis. Pass --a4c /path/to/A4C.mp4.")

    # 3. EchoPrime Suite
    if 'echoprime' in selected:
        print("\n[3/3] Running EchoPrime (21 structured metrics + report retrieval)...")
        cmd = [
            sys.executable, 'echoprime_cloud.py',
            '--input-root', str(patient_dir.parent if patient_dir.is_dir() else patient_dir),
            '--output-dir', str(out_dir / 'echoprime'),
            '--cache-dir', '/workspace/cache/echoprime' if os.path.exists('/workspace') else 'cache/echoprime'
        ]
        subprocess.run(cmd, check=False)

    print("\n" + "="*70)
    print(f"INFERENCE COMPLETE. Review JSONs and reports in {out_dir}")
    print("="*70)

if __name__ == '__main__':
    main()
