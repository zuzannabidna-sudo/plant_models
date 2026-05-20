# Multi-Model Plant Recognition & Cascaded Diagnostics (Proof of Concept)

This repository serves as a **Proof of Concept (PoC)** and an **inference evaluation baseline** for a cascaded plant analysis system. It provides four pre-trained core models alongside a unified CLI script designed to demonstrate how a modular, multi-stage machine learning pipeline can be implemented for production environments.

All core models are built upon the **EfficientNetB0** architecture, utilizing transfer learning from ImageNet, and are optimized for 224x224 RGB inputs.

---

## Core Concept: Modular Cascade Pipeline

Instead of training a single, massive monolithic model to handle both species and diseases simultaneously - which often suffers from cross-species symptom confusion and dataset imbalance. This project demonstrates a **modular cascade architecture**:

1. **Gatekeeper (Model 1):** A binary classifier (`leaf` vs. `no_leaf`) that filters incoming data. If no plant is detected with high confidence, the pipeline stops immediately, preventing invalid data from wasting downstream computing resources.
2. **Species Classifier (Model 2):** Identifies the plant variety across **48 distinct categories**.
3. **Dedicated Expert Models (Models 3.x):** Highly specialized networks trained strictly on the unique visual pathologies of a single specific genus (currently showcasing *Rose* and *Pothos*).

This modularity proves how easily an engineering system can scale: new species or diagnostics can be appended as independent blocks without retraining the existing ecosystem.

---

## Model Profiles & Evaluation Baseline

### 1. Model 1: Plant Detection Gatekeeper
* **Purpose:** Image verification and initial input filtration.
* **Dataset:** Curated imagery from the Unsplash platform.
* **Test Performance:**
  * **Accuracy / Precision / Recall:** 97.48%
  * **Area Under Curve (AUC):** 99.40%
  * **Loss:** 0.0931
* **Role in Pipeline:** High recall ensures valid plant images safely bypass the gatekeeper while filtering out noisy backgrounds.

### 2. Model 2: Houseplant Species Classification
* **Purpose:** Variety identification to determine the downstream diagnostic path.
* **Dataset Fusion:** Combined *House Plant Species* (Kaggle) and *Rose Leaf Disease Detection* datasets.
* **Test Performance:**
  * **Accuracy:** 85.43% | **Precision:** 91.18% | **Recall:** 80.26% | **AUC:** 99.21%
* **Human-in-the-Loop (HITL) Demonstration:** To demonstrate how to bridge the gap between model variance and high reliability in real-world apps, the provided script outputs **Top-3 Predictions**. In a full application, this allows the user to confirm the species before triggering medical diagnostics.

### 3. Model 3.1: Rose Pathology Expert
* **Purpose:** Diagnostic classification for the *Rose (Rosa L.)* genus.
* **Classes:** `Black spot`, `Downy mildew`, `Insects Infected`, `Mosaic`, and `Pure` (Healthy).
* **Test Performance:**
  * **Accuracy:** 77.99% | **Precision:** 84.04% | **Recall:** 69.10% | **AUC:** 94.51%
* **Note on Visual Noise:** The baseline highlights the challenges of real-world co-infections (e.g., a leaf displaying both mildew and insect bites), which introduces label noise but showcases robust probability ranking (94.51% AUC).

### 4. Model 3.2: Pothos Pathology Expert
* **Purpose:** Diagnostic classification for *Epipremnum aureum* (Pothos).
* **Classes:** `Bacterial wilt disease`, `Manganese Toxicity`, and `Healthy`.
* **Test Performance:**
  * **Accuracy / Precision / Recall:** 99.73% | **AUC:** 99.67% | **Loss:** 0.0812
* **Note on Visual Variance & Dataset Limitations:** * While the model demonstrates near-perfect separation on the test set due to the high visual variance between broad bacterial necroses, fine chemical toxicity spotting, and uniform healthy leaf textures, **it exhibits a distinct dataset limitation regarding cultivar variations (variegation).**
  * Because the training dataset lacked sufficient representation of variegated pothos cultivars (such as *Marble Queen*, *N-Joy*, or *Golden Pothos* with heavy yellow/white patterning), **the model incorrectly classifies natural, healthy leaf variegation as symptoms of disease or toxicity.** * This behavior perfectly benchmarks how a narrow feature distribution in training data can cause a model to misinterpret healthy genetic traits as visual anomalies (pathologies).
---
### 5. Data Validation Note: Resolving the "Money Plant" Nomenclature Ambiguity

A critical step in the data preparation phase involved resolving a significant botanical nomenclature ambiguity regarding the Pothos dataset:

* **The Issue:** The original dataset was published under the common regional name **"Money Plant"**. In Western botanical nomenclature, this term is highly ambiguous and typically refers to completely different species, such as *Pilea peperomioides* (Chinese Money Plant) or *Crassula ovata* (Jade Plant).
* **The Context:** In many Asian regions, however, "Money Plant" is the standard colloquial name for *Epipremnum aureum* (Pothos), deeply rooted in cultural associations with prosperity.
* **The Verification:** To ensure absolute data integrity, a manual visual morphology audit was conducted on the dataset's leaf structures. The analysis confirmed that the specimens are exclusively *Epipremnum aureum* (Pothos). 

This verification justified the inclusion of the dataset and underscores the necessity of strict domain-specific data auditing in machine learning pipelines, preventing cross-species target leakage.
---
## Datasets & Attribution

This project relies on several publicly available open-access datasets. We highly appreciate and acknowledge the invaluable contributions of the original researchers and authors who made their work available under Creative Commons licenses.

### 1. House Plant Species Dataset (Model 2 backbone)
* **Dataset Name:** House Plant Species
* **Author/Publisher:** KaKa (Kaggle)
* **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
* **Citation:** KaKa. (2023). *House Plant Species* [Dataset]. Kaggle. Available at: `https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species`

### 2. Rose Leaf Disease Detection Dataset (Model 2 expansion & Model 3.1)
* **Dataset Name:** Rose Leaf Disease Detection Dataset
* **Authors:** Rahat, Sakib Mahmud; Ahmed, Syed Sabbir; Mojumdar, Mayen Uddin (2024)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Repository:** Mendeley Data, V1
* **Digital Object Identifier (DOI):** [10.17632/sfxcjwf6cb.1](https://doi.org/10.17632/sfxcjwf6cb.1)

### 3. Advanced Dataset on Money Plant Diseases for AI Pathology Research (Model 3.2)
* **Dataset Name:** Advanced Dataset on Money Plant Diseases for AI Pathology Research
* **Author:** Ahmad, MD Hasan (2024)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Repository:** Mendeley Data, V3
* **Digital Object Identifier (DOI):** [10.17632/rzjww3vdxt.3](https://doi.org/10.17632/rzjww3vdxt.3)

### 4. Unsplash Imagery (Model 1 background data)
* **Source Platform:** Unsplash
* **License:** [Unsplash License](https://unsplash.com/license) (Free for commercial and non-commercial use, no attribution required but utilized here for pipeline design transparency).
