# Fine-Scale Permafrost and Soil Taxonomy Prediction

This repository contains code for fine-scale prediction of **permafrost presence/absence** and **soil taxonomy classification** across Alaska using both traditional machine learning and deep learning approaches.

## 📁 Repository Structure

- `RF/` — Implements a traditional **Random Forest** model for soil and permafrost prediction.
- `MISO/` — Implements **MISO**, a multimodal vision-based deep learning model that integrates integrates a pretrained geospatial foundation model based on the SWIN Transformer [1], implicit image functions for continuous spatial prediction [2], and contrastive learning for multimodal feature alignment and geo-location awareness.

## 🚀 Running Commands

### 🔧 Training

To train a model (e.g., on the AKSDB permafrost binary dataset with satellite and geospatial covariates):

```bash
CUDA_VISIBLE_DEVICES="0" python train.py \
    --config configs/aksdb_pf1m_bin/visual_geo__sat_cov__c19.yaml \
    --local_rank 0 \
    --master_port 18843 \
    --fold_id 0 \
    --split_mode kfold \
    --split_file data/kfold5_split_train_test_indices.json
```

### 🧪 Demo / Inference

To run inference on a specific grid tile:

```bash
python demo.py \
    --grid_id AK050H48V07 \
    --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__c19__sfold0/ \
    --interval 5 \
    --output_tif AK050H48V07_pf1m_interval5.tif
```

## 📌 Notes

- All models use Alaska’s soil observation dataset (AKSDB) and derive fine-scale predictions at 10-meter resolution.
- Inference outputs are saved as `.tif` files for spatial analysis or visualization in GIS software.
- The project supports k-fold cross-validation for robust evaluation.


## 📚 References

[1] Bastani, F., Wolters, P., Gupta, R., Ferdinando, J., & Kembhavi, A. (2023). Satlaspretrain: A large-scale dataset for remote sensing image understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

[2] Chen, Y., Liu, S., & Wang, X. (2021). Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.


