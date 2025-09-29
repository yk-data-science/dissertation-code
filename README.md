# Project Overview: Predicting Treatment Count and Visual Acuity from OCT Images

**Note:** This repository does not contain any medical images or case data due to privacy regulations.
Only the source code and documentation are provided.

## Goal

This project aims to develop deep learning models that can predict **the number of anti-VEGF treatments** and **visual acuity (VA)** outcomes **2 years after** the initial OCT scan. The models will use **baseline OCT B-scan images** as input.

- **Input:** Baseline OCT B-scan (grayscale) images  
  - For each of 14 cases (one case omitted due to unclear image):  
    - 1 central image (macula clearly visible)  
    - 6 peripheral images (lesion areas), verified with a clinician 
- **Output:** 
  - logs: accuracy
  - learning curve plots
  - Grad-Cam heatmap
  - Calculate confusion matrics and f1 score
- **Models:**  
  - Model 1 → Number of treatments after 2 years  
  - Model 2 → VA after 2 years  
- **Model type:**  
  - Baseline CNN (simple to prevent overfitting)  
  - Transfer learning CNN (MobileNetV2 due to small dataset)  

### Key Considerations

- **Data size:** Small number of labeled cases (14 cases)  
- **Artifacts:** 2 cases manually cleaned (no omission due to small dataset)  
- **Cross-validation:** 7-fold k-fold (7 splits)  
- **Augmentation:** Custom implementation (not library-based) to increase robustness  
- **Preprocessing:**  
  - Resize all images to a common small size, then crop/resize to 450×450 for Grad-CAM and transfer learning  
  - Augmentations include: horizontal flip, stretching, noise  
- **Explainability (XAI):** Grad-CAM heatmaps, compared with clinician evaluation  
  - Observed limitation: IRF and SRF not fully distinguishable  
- **Clinical metadata:** Not used in current phase; future multi-input models planned

- **comorbidity data:** Case A,L
- **manually edited data:**" Case A,H


---

## Project Phases

### Phase 1: Initial Model Using Selected Slices (Proof of Concept: PoC)

| Item                 | Description                                  |
|----------------------|----------------------------------------------|
| **Data**             | 14 cases (1 case omitted due to unclear image), 1 central + 6 peripheral images per case |
| **Preprocessing**    | Resize and normalize; 450×450 for Grad-CAM/transfer learning; artifacts manually removed |
| **Artifacts**        | Manually removed for 2 cases             |
| **Model**            | Baseline CNN (simple architecture to prevent overfitting) |
| **Learning Strategy**| 7-fold k-fold cross-validation              |
| **Transfer Learning**| MobileNetV2 used due to small dataset       |
| **Label**            | Binary classification:<br>- Treatment count: `<14` / `≥14`<br>- VA: `<72` / `≥72` |
| **XAI:**            | Grad-CAM heatmaps; improve interpretability |

---

### Phase 2+ Future Extension: Expanded Dataset and Improved Training

- **Data:** Increase number of images (≥50), including unlabeled cases  
- **Possible methods:**
  - Use semi-supervised or unsupervised learning for unlabeled data  
  - Data balancing (under-sampling or over-sampling)  
  - Tune CNN architecture (layers, filters, etc.)  
  - Apply stronger augmentation  
 


| Direction | Description |
|----------|-------------|
| **More labels** | Add multi-class treatment labels; predict VA as regression |
| **Additional inputs** | Include clinical metadata (age, sex, comorbidities) for multi-input models |
| **Prediction targets** | Predict treatment frequency and VA using different models |
| **Preprocessing automation** | Automate cropping, alignment, ROI extraction |

---

## Final Objective

Create a clinically meaningful AI system that:
- Predicts long-term treatment need and VA
- Is robust to image noise and variation
- Is explainable for use in ophthalmology

## Project Structure

The project directory is organized as follows:

```
PREDICT/
├─ data/        # Preprocessed input images (not uploaded here due to regulations)
├─ output/      # Learning curves, Grad-CAM visualizations, logs
├─ src/
│   ├─ main_treatment_cnn.py
│   └─ main_va_cnn.py
└─ requirements.txt
```

- `data/` contains preprocessed OCT images for training and evaluation.  
- `output/` stores model outputs, logs, and visualization results.  
- `src/` includes main scripts, model definitions, dataset handling, and augmentation utilities.  
- `requirements.txt` lists all Python packages required for the project.

## Notes
- Code organization: Due to time constraints, the current implementation has not yet been fully modularized.  
- Ideally, functions and classes would be separated by functionality, but for now, most of the code resides in a single script.
(Ideal structure is in src/codes/ folder.)

## Submission
The complete project is submitted as a **Git repository**. 


### Requirements

All required Python packages are listed in the repository's [`requirements.txt`](./requirements.txt).  
To set up the environment, run:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

# Install packages
pip install -r requirements.txt
