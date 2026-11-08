"""Generate decoded RF-DETR pose reference from a real LiteRT artifact.

Uses the upstream PyTorch postprocessor on identical raw output tensors, testing
confidence fusion as well as geometry. Run native IntegrationTests with
RFDETR_POSE_REFERENCE set to the output directory. The official background-first
COCO preview schema (0,17) is intentional.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    from ai_edge_litert.interpreter import Interpreter
    from rfdetr.models.postprocess import PostProcess

    interpreter = Interpreter(model_path=str(args.model), num_threads=1)
    interpreter.allocate_tensors()
    (detail,) = interpreter.get_input_details()
    tensor = np.random.default_rng(0).standard_normal(detail['shape']).astype(np.float32)
    interpreter.set_tensor(detail['index'], tensor)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(value['index']) for value in interpreter.get_output_details()]
    boxes, logits, keypoints = map(torch.from_numpy, outputs)
    if logits.shape[-1] != 2 or keypoints.shape[-2:] != (34, 8):
        raise ValueError('Expected official background-first COCO pose schema')
    postprocess = PostProcess(num_select=boxes.shape[1], num_keypoints_per_class=[0, 17], trace_alpha=0.2)
    scores, labels, queries = postprocess._select_topk(logits[..., 1:])
    sizes = torch.tensor([[480, 640]])
    selected_boxes = postprocess._gather_and_scale_boxes(boxes, queries, sizes)
    decoded = postprocess._postprocess_keypoints(keypoints, scores, labels + 1, selected_boxes, queries, sizes)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, output in enumerate(outputs):
        output.tofile(args.output_dir / f'output_{i}.bin')
    rows = torch.cat((decoded['scores'][:, None], decoded['boxes'], decoded['keypoints'][..., :2].flatten(1)), dim=1)
    np.savetxt(args.output_dir / 'expected.txt', rows.numpy(), fmt='%.9g')
    (args.output_dir / 'metadata.txt').write_text(f'{boxes.shape[1]} {detail["shape"][2]}\n')
    (args.output_dir / 'labels.txt').write_text('person\n')
    print(f'{len(rows)} upstream-decoded poses written to {args.output_dir}')


if __name__ == '__main__':
    main()
