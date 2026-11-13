import os, sys, json, csv
from pathlib import Path

def resynthesize():
    results_dir = Path('/workspace/results')
    patients_dir = Path('/workspace/patients')
    patient_dirs = sorted([p for p in patients_dir.iterdir() if p.is_dir() and not p.name.startswith('.')])
    
    sr_file = Path('/workspace/extracted_sr_clinical.json')
    sr_gt = json.loads(sr_file.read_text()) if sr_file.exists() else {}

    cohort = []
    for p_dir in patient_dirs:
        p_name = p_dir.name
        row = {
            'patient_name': p_name,
            # Ground Truth from SR
            'ground_truth_ef': None,
            'ground_truth_lvedv': None,
            'ground_truth_lvesv': None,
            'ground_truth_lvidd': None,
            'ground_truth_e_a': None,
            'ground_truth_e_e_prime': None,
            # Ejection Fraction Models
            'panecho_ef': None,
            'echonet_ef': None,
            'echoclip_ef': None,
            'echoprime_ef': None,
            # Strain Models (GLS)
            'panecho_gls': None,
            'unity_gls': None,
            'echonet_gls': None,
            # Chamber Dimensions & Hemodynamics
            'panecho_lvedv': None,
            'panecho_lvesv': None,
            'panecho_lvidd': None,
            'panecho_rvsp': None,
            'echoprime_pasp': None,
            'panecho_tapse': None,
            # Valvular & Pathology Flags
            'panecho_mr_prob': None,
            'echoprime_mr_flag': None,
            'panecho_as_prob': None,
            'panecho_effusion_prob': None,
            'echoprime_effusion_flag': None
        }

        # 1. Ground Truth Matching
        for gt_pname, gt_vals in sr_gt.items():
            clean_gt = gt_pname.replace(' ', '_').replace('^', '_')
            if clean_gt in p_name or p_name.startswith(clean_gt[:10]):
                meas = gt_vals.get('measurements', {})
                for k, v in meas.items():
                    k_low = k.lower()
                    if 'ejection fraction' in k_low:
                        row['ground_truth_ef'] = v['value']
                    elif 'end diastolic volume' in k_low:
                        row['ground_truth_lvedv'] = round(v['value'] / 1000.0, 1) # mm3 to mL
                    elif 'end systolic volume' in k_low:
                        row['ground_truth_lvesv'] = round(v['value'] / 1000.0, 1)
                    elif 'internal end diastolic dimension' in k_low:
                        row['ground_truth_lvidd'] = v['value']
                    elif 'mitral valve e to a ratio' in k_low:
                        row['ground_truth_e_a'] = v['value']
                    elif 'ratio of mv peak velocity to lv peak tissue velocity' in k_low:
                        row['ground_truth_e_e_prime'] = v['value']

        # 2. PanEcho
        pan_p = results_dir / p_name / 'panecho.json'
        if pan_p.exists():
            try:
                data = json.loads(pan_p.read_text())
                preds = data.get('predictions', {})
                if 'EF' in preds: row['panecho_ef'] = round(preds['EF']['value'], 1)
                if 'GLS' in preds: row['panecho_gls'] = round(preds['GLS']['value'], 1)
                if 'LVEDV' in preds: row['panecho_lvedv'] = round(preds['LVEDV']['value'], 1)
                if 'LVESV' in preds: row['panecho_lvesv'] = round(preds['LVESV']['value'], 1)
                if 'LVIDd' in preds: row['panecho_lvidd'] = round(preds['LVIDd']['value'] * 10.0, 1)
                if 'RVSP' in preds: row['panecho_rvsp'] = round(preds['RVSP']['value'], 1)
                if 'TAPSE' in preds: row['panecho_tapse'] = round(preds['TAPSE']['value'], 2)
                if 'MVRegurgitation' in preds:
                    probs = preds['MVRegurgitation'].get('probabilities', {})
                    row['panecho_mr_prob'] = round(probs.get('Moderate|Severe', 0.0) + probs.get('Mild', 0.0), 2)
                if 'AVStenosis' in preds:
                    probs = preds['AVStenosis'].get('probabilities', {})
                    row['panecho_as_prob'] = round(probs.get('Severe', 0.0) + probs.get('Mild|Moderate', 0.0), 2)
                if 'pericardial-effusion' in preds:
                    row['panecho_effusion_prob'] = round(preds['pericardial-effusion'].get('probability', 0.0), 2)
            except Exception as e:
                print(f"Error parsing PanEcho for {p_name}: {e}")

        # 3. EchoNet-Dynamic
        echo_p = results_dir / p_name / 'echonet.json'
        if echo_p.exists():
            try:
                data = json.loads(echo_p.read_text())
                ef = data.get('predictions', {}).get('ef_percent')
                if ef is not None: row['echonet_ef'] = round(ef, 1)
            except Exception as e:
                print(f"Error parsing EchoNet for {p_name}: {e}")

        # 4. EchoCLIP
        clip_p = results_dir / p_name / 'echoclip.json'
        if clip_p.exists():
            try:
                data = json.loads(clip_p.read_text())
                ef = data.get('predictions', {}).get('ef_percent')
                if ef is not None: row['echoclip_ef'] = round(ef, 1)
            except Exception as e:
                print(f"Error parsing EchoCLIP for {p_name}: {e}")

        # 5. EchoPrime
        ep_p = results_dir / 'echoprime' / p_name / 'echoprime.json'
        if ep_p.exists():
            try:
                data = json.loads(ep_p.read_text())
                preds = data.get('predictions', {})
                if 'ejection_fraction' in preds: row['echoprime_ef'] = round(preds['ejection_fraction'], 1)
                if 'pulmonary_artery_pressure_continuous' in preds: row['echoprime_pasp'] = round(preds['pulmonary_artery_pressure_continuous'], 1)
                if 'mitral_regurgitation' in preds: row['echoprime_mr_flag'] = round(preds['mitral_regurgitation'], 2)
                if 'pericardial_effusion' in preds: row['echoprime_effusion_flag'] = round(preds['pericardial_effusion'], 2)
            except Exception as e:
                print(f"Error parsing EchoPrime for {p_name}: {e}")

        # 6. Unity-GLS
        u_p = results_dir / p_name / 'strain' / 'unity_gls' / 'result.json'
        if u_p.exists():
            try:
                data = json.loads(u_p.read_text())
                gls = data.get('signed_longitudinal_contour_shortening_percent')
                if gls is not None: row['unity_gls'] = round(gls, 1)
            except Exception as e:
                print(f"Error parsing Unity-GLS for {p_name}: {e}")

        # 7. EchoNet-Strain
        es_p = results_dir / p_name / 'strain' / 'echonet_strain' / 'result.json'
        if es_p.exists():
            try:
                data = json.loads(es_p.read_text())
                gls = data.get('signed_longitudinal_contour_shortening_percent')
                if gls is not None: row['echonet_gls'] = round(gls, 1)
            except Exception as e:
                print(f"Error parsing EchoNet-Strain for {p_name}: {e}")

        cohort.append(row)

    # Save
    master_json = results_dir / 'master_cohort_interpretation.json'
    master_json.write_text(json.dumps(cohort, indent=2))
    print(f"Updated Master JSON: {master_json}")

    master_csv = results_dir / 'master_cohort_summary.csv'
    if cohort:
        keys = list(cohort[0].keys())
        with open(master_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(cohort)
        print(f"Updated Master CSV: {master_csv}")

if __name__ == '__main__':
    resynthesize()
