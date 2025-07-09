## Configuration File Naming Convention

The configuration files in "configs/pretrain" follow this naming pattern:
`yolo_world_[version]_[size]_[text_encoder]_[neck]_[normalization]_[special_features]_[learning_rate]_[epochs]_[gpu_config]_[datasets]_[training_type]_[resolution]_[eval_dataset].py`

### Component Breakdown:

1. **Model Version**: `v2` - Indicates YOLO-World version 2

2. **Model Size**: `s`, `m`, `l`, `x`, `xl` - Represents Small, Medium, Large, eXtra-large, and eXtra-Large variants respectively

3. **Text Encoder**: 
   - Default: CLIP-Base (no additional notation)
   - Enhanced: `clip_large` - Uses CLIP-Large text encoder for better performance

4. **Neck Architecture**: `vlpan` - Vision-Language Path Aggregation Network

5. **Normalization**: `bn` - Batch Normalization

6. **Special Features** (optional):
   - `noeinsum` - Indicates no einsum operations are used
   - `efficient_neck` - Uses efficient network architecture

7. **Learning Rate**: `2e-3` - Base learning rate (2×10^-3)

8. **Training Epochs**: `100e` - Number of training epochs (100 for pre-training)

9. **GPU Configuration**: `4x8gpus` - 4 nodes × 8 GPUs = 32 GPUs total for distributed training

10. **Training Datasets**: 
    - `obj365v1_goldg` - Objects365v1 + GoldG datasets (primary combination)
    - `obj365v1_goldg_cc3mlite` - Adds CC3M-Lite dataset

11. **Training Type**: `train` - Indicates pre-training configuration

12. **Fine-tuning Resolution** (optional):
    - `800ft` - Fine-tuning at 800×800 resolution
    - `1280ft` - Fine-tuning at 1280×1280 resolution

13. **Evaluation Dataset**: 
    - `lvis_minival` - LVIS minival dataset for evaluation
    - `lvis_val` - LVIS validation dataset for evaluation
