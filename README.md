# 💡 Image-Based Lighting & HDR Fusion  
**CS445: Computational Photography — Project 4**  
📅 **Due Date:** April 16, 2025  
👨‍💻 **Author:** Taha Wasiq  
📓 **Notebook:** `Image_Based_Lighting.ipynb`

---

## 🧠 Overview

This project implements a full **High Dynamic Range (HDR)** imaging and **Image-Based Lighting (IBL)** pipeline using multiple exposure images of a spherical mirror. The goal is to reconstruct HDR radiance maps, convert them to usable lighting environments, and use them to relight and composite synthetic 3D objects into real-world scenes using Blender.

Core techniques include:
- HDR reconstruction from LDR sequences
- Estimation of the camera response function
- Mirror ball to panoramic (equirectangular) conversion
- Image-based lighting with Blender (Cycles)
- Evaluation of irradiance consistency and dynamic range

---

## 📂 Project Structure

```text
/
├── Image_Based_Lighting.ipynb         # ✔️ Main notebook (HDR + relighting)
├── utils.py                           # ✔️ Provided utility functions and helpers
├── render1.png                        # ✔️ Rendered result using Blender
├── images/                            # ❌ Not included — download from Drive, Includes rendered images, with final result
├── samples/                           # ❌ Not included — download from Drive, Includes sample images
└── README.md                          # ✔️ This file


---


## 📦 Dataset & Folder Access

> ⚠️ **Note:** Due to GitHub’s file size limits, the `images/` and `samples/` folders have been excluded from this repository. These folders contain:
>
> - Raw LDR images of a spherical mirror at multiple exposures  
> - Background photo for 3D object insertion  
> - Sample HDR content for testing  
>
> You can download the full dataset from the links below:

👉 **Google Drive Folders:**
- 📁 [images/](https://drive.google.com/drive/folders/1aMLa-tkiSDGyYyKx99UXLiP9YUvlTVrS?usp=sharing)
- 📁 [samples/](https://drive.google.com/drive/folders/12knvUzpd9x3lESZyj4O4kSLyaI4wpOOF?usp=sharing)


> After downloading, place both folders in the root directory of the repo so the notebook can locate them correctly.

---

## 🚀 Implemented Features

### ✅ HDR Radiance Map Reconstruction (70 pts)
- **Naive Merging**: Average irradiance estimate from LDRs
- **Weighted Merging**: Exposure-aware pixel weighting
- **CRF Estimation**: Estimated camera response function using Debevec's method
- **HDR Output**: Visualized and saved final HDR result (with optional tone mapping)

### ✅ Equirectangular Projection (10 pts)
- Converted mirror ball images to panoramic HDR format for rendering
- Used reflection vector math and provided `get_equirectangular_image()` function

### ✅ Image-Based Lighting in Blender (30 pts)
- Used custom HDR panoramas as environment maps
- Inserted and relit synthetic objects using Blender Cycles

---

## 📈 Evaluation Metrics

The notebook includes:
- 📉 Consistency check for irradiance estimates across exposure levels
- 🌈 Dynamic range measurement of output HDR images
- 📊 Comparison of naive, weighted, and calibrated (CRF) methods

---

## 🧰 Dependencies

Install required libraries with:

```bash
pip install numpy scipy opencv-python matplotlib imageio
