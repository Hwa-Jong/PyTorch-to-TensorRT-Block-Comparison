import os
import argparse
from python.save_model import load_opt_from_json, export_to_onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to ONNX format"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
        choices=[64, 256, 1024],
        help="Input size (default: 64, choices: 64, 256, 1024)",
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable FP16 mode (default: false, choices: true, false)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help='Model name to export (e.g., exp1_resnet.json) or "all" to export all models (default: all)',
    )
    parser.add_argument(
        "--opts_path",
        type=str,
        default=os.path.join("python", "opts"),
        help="Path to options directory (default: python/opts)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path = args.opts_path

    if args.model == "all":
        names = os.listdir(path)
    else:
        names = [args.model]

    for name in names:
        if not name.endswith(".json"):
            continue
        opt = load_opt_from_json(os.path.join(path, name))
        opt.name = name
        fp16_bool = args.fp16.lower() == "true"
        print(
            f"Exporting {name} with input_size={args.input_size}, fp16={fp16_bool}"
        )
        export_to_onnx(opt, input_size=args.input_size, fp16=fp16_bool)
