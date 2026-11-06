"""Generate native-backend parity inputs/outputs using the Python LiteRT interpreter.

Usage: python tests/integration/generate_litert_reference.py MODEL.tflite OUTPUT_DIR
Run integration_tests with RFDETR_LITERT_MODEL and RFDETR_LITERT_REFERENCE set.
Requires ai-edge-litert at LITERT_VERSION (versions.env) and numpy.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(args.model), num_threads=1)
    interpreter.allocate_tensors()
    (input_detail,) = interpreter.get_input_details()
    tensor = np.random.default_rng(0).standard_normal(input_detail['shape']).astype(np.float32)
    interpreter.set_tensor(input_detail['index'], tensor)
    interpreter.invoke()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor.tofile(args.output_dir / 'input.bin')
    for index, output in enumerate(interpreter.get_output_details()):
        value = interpreter.get_tensor(output['index'])
        value.tofile(args.output_dir / f'output_{index}.bin')
        print(index, output['name'], value.shape)


if __name__ == '__main__':
    main()
