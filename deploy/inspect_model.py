# Quick script to inspect the RF-DETR model structure
import torch
from rfdetr import RFDETRSegPreview

model_wrapper = RFDETRSegPreview()
print(f"model_wrapper type: {type(model_wrapper)}")
print(f"model_wrapper.model type: {type(model_wrapper.model)}")
print(f"\nmodel_wrapper attributes: {[a for a in dir(model_wrapper) if not a.startswith('_')]}")
print(f"\nmodel_wrapper.model attributes: {[a for a in dir(model_wrapper.model) if not a.startswith('_')]}")

# Check for the actual PyTorch model
if hasattr(model_wrapper.model, 'module'):
    print(f"\nmodel_wrapper.model.module type: {type(model_wrapper.model.module)}")
    print(f"model_wrapper.model.module attributes: {[a for a in dir(model_wrapper.model.module) if not a.startswith('_')][:20]}")
