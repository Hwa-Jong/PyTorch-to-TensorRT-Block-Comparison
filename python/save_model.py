import argparse
import json
import os
import torch
import torch.nn as nn

from ptflops import get_model_complexity_info
from models.classification import BaseModle


def export_to_onnx(opt, input_size=256, fp16=True):
    if not os.path.exists("onnx"):
        os.makedirs("onnx")

    name = os.path.splitext(opt.name)[0]
    block_type = opt.block_type
    channels = opt.channels
    num_blocks = opt.num_blocks
    norm_type = opt.norm_type
    act_type = opt.act_type
    num_classes = opt.num_classes

    save_path = os.path.join("onnx", f"{name}.onnx")

    model = BaseModle(
        block_type, channels, num_blocks, norm_type, act_type, num_classes
    )
    macs, params = get_model_complexity_info(
        model,
        (3, input_size, input_size),
        as_strings=True,
        print_per_layer_stat=False,
    )

    model.eval()

    input_names = ["input_1"]
    output_names = ["output_1"]
    dummy_input = torch.randn(1, 3, input_size, input_size)

    if fp16:
        model = model.half()
        dummy_input = dummy_input.half()

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        verbose=True,
    )

    opt.input_size = input_size
    opt.fp16 = fp16
    opt.params = params
    opt.macs = macs

    opt_save_path = os.path.join("onnx", f"{name}.json")
    with open(opt_save_path, "w") as f:
        json.dump(vars(opt), f, indent=4)
    print(f"Options saved to {opt_save_path}")


def load_opt_from_json(json_path):
    with open(json_path, "r") as f:
        opt_dict = json.load(f)
    return argparse.Namespace(**opt_dict)


if __name__ == "__main__":
    path = os.path.join("python", "opts")
    names = os.listdir(path)

    input_size = 1024
    fp16_mode = False

    for name in names:
        opt = load_opt_from_json(os.path.join(path, name))
        opt.name = name
        export_to_onnx(opt, input_size=input_size, fp16=fp16_mode)
