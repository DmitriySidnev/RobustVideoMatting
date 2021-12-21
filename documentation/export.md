# Expor to ONNX

To export a model to ONNX follow the steps below:

1. Train a model or download existing weights
2. Run the export script `export.py`

Example for the pretrained model with `mobilenetv3` backbone:

```bash
cd RobustVideoMatting
pip install -r requirements_inference.txt
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
python export.py \
  --backbone mobilenetv3 \
  --checkpoint rvm_mobilenetv3.pth \
  --output model.onnx \
  --validate
```
