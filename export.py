import argparse
import torch
from model import MattingNetwork

parser = argparse.ArgumentParser(description='Export to ONNX')

parser.add_argument('--backbone', type=str, required=True, choices=['resnet50', 'mobilenetv3'])
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--validate', action='store_true')

args = parser.parse_args()


model = MattingNetwork(variant=args.backbone).eval().cuda()
model.load_state_dict(torch.load(args.checkpoint))

src_shape = (1, 3, 720, 1280)

rand_input = torch.rand(src_shape, dtype=torch.float32).cuda()
rec = [None] * 4
downsample_ratio = 0.4

fgr, pha, *rec = model(rand_input, *rec, downsample_ratio)

rec = [r * 0. for r in rec]

# Export ONNX
input_names=['src', 'r1', 'r2', 'r3', 'r4']
output_names = ['fgr', 'pha', 'rr1', 'rr2', 'rr3', 'rr4']

dynamic_axes = {
    'src': {0: 'batch', 2: 'height', 3: 'width'},
    'r1': {0: 'batch', 2: 'r1_height', 3: 'r1_width'},
    'r2': {0: 'batch', 2: 'r2_height', 3: 'r2_width'},
    'r3': {0: 'batch', 2: 'r3_height', 3: 'r3_width'},
    'r4': {0: 'batch', 2: 'r4_height', 3: 'r4_width'},
}

torch.onnx.export(
    model=model,
    args=(rand_input, *rec),
    f=args.output,
    verbose=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes)

print(f'ONNX model saved at: {args.output}')

# Validation
if args.validate:
    import onnxruntime

    print(f'Validating ONNX model.')

    src = torch.randn(*src_shape).cuda()

    with torch.no_grad():
        out_torch = model(src, *rec)

    sess = onnxruntime.InferenceSession(args.output)
    out_onnx = sess.run(None, {
        'src': src.cpu().numpy(),
        'r1': rec[0].detach().cpu().numpy(),
        'r2': rec[1].detach().cpu().numpy(),
        'r3': rec[2].detach().cpu().numpy(),
        'r4': rec[3].detach().cpu().numpy(),
    })

    e_max = 0
    for a, b, name in zip(out_torch, out_onnx, output_names):
        b = torch.as_tensor(b)
        e = torch.abs(a.cpu() - b).max()
        e_max = max(e_max, e.item())
        print(f'"{name}" output differs by maximum of {e}')

    if e_max < 0.01:
        print('Validation passed.')
    else:
        raise 'Validation failed.'

