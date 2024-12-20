import argparse
import json
import os
import torch
import torch.nn as nn

from ptflops import get_model_complexity_info
from models.classification import BaseModle


def export_to_onnx(opt):
    if not os.path.exists("onnx"):
        os.makedirs("onnx")

    name = opt.name
    block_type = opt.block_type
    channels = opt.channels
    num_blocks = opt.num_blocks
    norm_type = opt.norm_type
    act_type = opt.act_type
    num_classes = opt.num_classes
    input_size = opt.input_size

    save_path = os.path.join("onnx", f"{os.path.splitext(name)[0]}.onnx")

    model = BaseModle(
        block_type, channels, num_blocks, norm_type, act_type, num_classes
    )
    mmac, params = get_model_complexity_info(
        model,
        (3, input_size, input_size),
        as_strings=True,
        print_per_layer_stat=False,
    )

    model.eval()

    input_names = ["input_1"]
    output_names = ["output_1"]
    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        verbose=True,
    )

    opt.params = params
    opt.mmac = mmac

    opt_save_path = os.path.join("onnx", f"{name}_opt.json")
    with open(opt_save_path, "w") as f:
        json.dump(vars(opt), f, indent=4)
    print(f"Options saved to {opt_save_path}")


def load_opt_from_json(json_path):
    with open(json_path, "r") as f:
        opt_dict = json.load(f)
    return argparse.Namespace(**opt_dict)


if __name__ == "__main__":
    path = os.path.join("python", "opts", "same_Params")
    names = os.listdir(path)
    for name in names:
        opt = load_opt_from_json(os.path.join(path, name))
        opt.name = name
        export_to_onnx(opt)
