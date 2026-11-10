# Echo Deep Learning Suite: Complete Docker Container

A unified, GPU-accelerated container integrating the **6 leading echocardiography deep learning foundation models** for automated clinical assessment, chamber quantification, global longitudinal strain (GLS), ejection fraction (LVEF), and full clinical report retrieval.

Container Registry: `ghcr.io/medstats/echo-deep-learning-docker:latest`  
Vast.ai Template: **`Echo-Deep-Learning-GPU`** (Template ID: `798717`)

---

## 1. Integrated Models & Capabilities

| Model | Architecture | Primary Clinical Outputs | Required / Useful Views |
| :--- | :--- | :--- | :--- |
| **PanEcho** | Multi-task 3D ConvNet | **40 clinical parameters**:<br>• LV function/volumes: `EF`, `GLS`, `LVEDV`, `LVESV`, `LVSV`, `LVWallMotionAbnormalities`<br>• Geometry/Thickness: `IVSd`, `LVPWd`, `LVIDd`, `LVIDs`, `LVOTDiam`, `LVSize`<br>• Diastology/Hemodynamics: `LVDiastolicFunction`, `E\|EAvg`, `RVSP`, `RAP-8-or-higher`<br>• Right Ventricle/Atria: `RVSize`, `RVSystolicFunction`, `TAPSE`, `LASize`, `RASize`<br>• Valves/Pericardium: `AVStenosis`, `AVPkVel`, `MVRegurgitation`, `TVRegurgitation`, `pericardial-effusion` | Single or multi-view MP4 (PLAX, PSAX, A4C, A2C, A3C) |
| **EchoPrime** | Multi-View Vision Foundation Model (1.2M candidate bank) | **21 structured phenotypes** (EF, RV function, Atrial dilation, Valve phenotypes, PA pressure) + **Similarity-Retrieved Structured Clinical Report** | Multi-view patient cine study (PLAX + Apical) |
| **EchoNet-Dynamic** | 3D ResNet (R2+1D) | Supervised **Left Ventricular Ejection Fraction (LVEF %)** regression | Verified Apical 4-Chamber (A4C) |
| **EchoCLIP** | Vision-Language Foundation Model | Zero-shot EF estimation, pacemaker/device detection, and frame-text similarity embeddings | Multi-view cine frames |
| **Unity-GLS** | HRNetV2M10 Landmark Tracking | **Global Longitudinal Strain (GLS %)** via frame-by-frame endocardial contour tracking | Apical views (A4C, A2C, A3C) |
| **EchoNet-Strain** | DeepLabV3-ResNet50 | Automated LV segmentation masks and **contour length shortening (%)** across cardiac cycles | Apical 4-Chamber (A4C) |

---

## 2. Input Folder Structure

To run inference, organize your patient cines into folders by case name:

```text
patients/
├── CASE_001/
│   ├── A4C.mp4          # Apical 4-Chamber (used by EchoNet, Strain, PanEcho, EchoPrime)
│   ├── PLAX.mp4         # Parasternal Long Axis (used by PanEcho, EchoPrime)
│   └── PSAX.mp4         # Parasternal Short Axis (used by PanEcho, EchoPrime)
└── CASE_002/
    ├── video_01_A4C.mp4
    └── video_02_PLAX.mp4
```

> [!TIP]
> **View Labeling Rule:** If a file contains `A4C` or `4CH` in its filename, `run_all_echo.py` and `run_strain.py` automatically route it to the strain and EchoNet pipelines.

---

## 3. How to Elicit Inference

### Option A: The Master 1-Command Runner (Runs All Models)
Execute all 6 models sequentially on a patient study:

```bash
python run_all_echo.py --patient patients/CASE_001 --output results/CASE_001
```

To run only specific models:
```bash
python run_all_echo.py --patient patients/CASE_001 --output results/CASE_001 --models panecho unity_gls echoprime
```

---

### Option B: Running Individual Models

#### 1. PanEcho (40 Clinical Outputs)
```bash
python run_inference.py --patient patients/CASE_001 --models panecho --output results/CASE_001
```
* **Output:** `results/CASE_001/panecho.json` containing numeric predictions for EF, GLS, wall motion, dimensions, and valve grading.

#### 2. EchoNet-Dynamic (LVEF %)
```bash
python run_inference.py --patient patients/CASE_001 --models echonet --output results/CASE_001
```
* **Output:** `results/CASE_001/echonet.json` with direct LVEF regression (e.g. `{"ejection_fraction": 54.2}`).

#### 3. EchoCLIP (Zero-Shot & Device Similarity)
```bash
python run_inference.py --patient patients/CASE_001 --models echoclip --output results/CASE_001
```
* **Output:** `results/CASE_001/echoclip.json` with zero-shot EF rankings and pacemaker device similarity scores.

#### 4. Unity-GLS (Global Longitudinal Strain)
```bash
python run_strain.py --models unity_gls --input patients/CASE_001/A4C.mp4 --view A4C --output results/CASE_001/strain
```
* **Output:** `results/CASE_001/strain/unity_gls/result.json` containing:
  * `signed_longitudinal_contour_shortening_percent`: e.g. `-18.2%`
  * Length curve across all frames
  * Contour overlay preview video

#### 5. EchoNet-Strain (DeepLabV3 LV Segmentation & Strain)
```bash
python run_strain.py --models echonet_strain --input patients/CASE_001/A4C.mp4 --view A4C --output results/CASE_001/strain
```
* **Output:** `results/CASE_001/strain/echonet_strain/result.json` with segmentation masks and cycle shortening percent.

#### 6. EchoPrime (21 Phenotypes + Retrieved Report)
```bash
python echoprime_cloud.py --input-root patients --output-dir results/echoprime
```
* **Output:**
  * `results/echoprime/CASE_001/echoprime.json`: 21 structured metrics (EF, RV function, atrial sizes, PA pressure).
  * `results/echoprime/CASE_001/retrieved_report.txt`: Complete structured clinical echocardiography report matching the study embeddings.

---

## 4. Running on Cloud Platforms

### Running on Vast.ai
1. In the Vast.ai console, select your template **`Echo-Deep-Learning-GPU`** (Template ID: `798717`).
2. Rent any RTX 3090, RTX 4090, or A5000 GPU instance.
3. Once booted, SSH into the machine:
   ```bash
   cd /workspace
   python run_all_echo.py --patient patients/ECHO_01 --output results/ECHO_01
   ```

### Running on Google Colab or Kaggle (1 Cell)
```python
# In a Colab or Kaggle GPU notebook cell:
!git clone https://github.com/medstats/echo-deep-learning-docker.git
%cd echo-deep-learning-docker
!pip install -r repos/PanEcho/requirements.txt open_clip_torch timm pydicom transformers==4.57.0
!python run_all_echo.py --patient /content/my_patient_study --output /content/results
```

---

## 5. Clinical Review & Interpretation Rules

* **Speckle Tracking vs Contour Shortening:** Unity-GLS and EchoNet-Strain measure 2D endocardial contour length changes, not 3D ultrasound speckle-tracking. Report single-view shortening accurately as single-plane contour GLS.
* **PanEcho GLS vs Measured GLS:** PanEcho's `GLS` key is a statistical regression estimate, not a measured length curve. For measured strain, refer to the Unity-GLS output.
* **EchoNet-Dynamic View Gate:** Always provide an authentic Apical 4-Chamber (A4C) cine loop. Never pass PLAX or PSAX to EchoNet-Dynamic.
* **EchoPrime Reports:** Retrieved report sections are based on cosine similarity against 1.2M historical reports. They provide high-value clinical drafting but require review by a credentialed clinician.
