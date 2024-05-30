# Copyright 2023 Radboud University Medical Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified version of the codes written by Rina Bao (rina.bao@childrens.harvard.edu)
# for BONBID-HIE MICCAI Challenge 2023 (https://bonbid-hie2023.grand-challenge.org/).


import json
from dataclasses import dataclass, make_dataclass
from enum import Enum
from typing import Any, Dict
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import SimpleITK
import os
from scipy import ndimage

from data_utils import create_input_array

INPUT_PREFIX = Path("/input")
OUTPUT_PREFIX = Path("/output")


class IOKind(str, Enum):
    JSON = "JSON"
    IMAGE = "Image"
    FILE = "File"


class InterfaceKind(str, Enum):
    # TODO taken from https://github.com/comic/grand-challenge.org/blob/ffbae21af534caed9595d9bc48708c5f753b075c/app/grandchallenge/components/models.py#L69
    #      would be better to get this directly from the schema

    def __new__(cls, value, annotation, io_kind):
        member = str.__new__(cls, value)
        member._value_ = value
        member.annotation = annotation
        member.io_kind = io_kind
        return member

    STRING = "String", str, IOKind.JSON
    INTEGER = "Integer", int, IOKind.JSON
    FLOAT = "Float", float, IOKind.JSON
    BOOL = "Bool", bool, IOKind.JSON
    ANY = "Anything", Any, IOKind.JSON
    CHART = "Chart", Dict[str, Any], IOKind.JSON

    # Annotation Types
    TWO_D_BOUNDING_BOX = "2D bounding box", Dict[str, Any], IOKind.JSON
    MULTIPLE_TWO_D_BOUNDING_BOXES = "Multiple 2D bounding boxes", Dict[str, Any], IOKind.JSON
    DISTANCE_MEASUREMENT = "Distance measurement", Dict[str, Any], IOKind.JSON
    MULTIPLE_DISTANCE_MEASUREMENTS = "Multiple distance measurements", Dict[str, Any], IOKind.JSON
    POINT = "Point", Dict[str, Any], IOKind.JSON
    MULTIPLE_POINTS = "Multiple points", Dict[str, Any], IOKind.JSON
    POLYGON = "Polygon", Dict[str, Any], IOKind.JSON
    MULTIPLE_POLYGONS = "Multiple polygons", Dict[str, Any], IOKind.JSON
    LINE = "Line", Dict[str, Any], IOKind.JSON
    MULTIPLE_LINES = "Multiple lines", Dict[str, Any], IOKind.JSON
    ANGLE = "Angle", Dict[str, Any], IOKind.JSON
    MULTIPLE_ANGLES = "Multiple angles", Dict[str, Any], IOKind.JSON
    ELLIPSE = "Ellipse", Dict[str, Any], IOKind.JSON
    MULTIPLE_ELLIPSES = "Multiple ellipses", Dict[str, Any], IOKind.JSON

    # Choice Types
    CHOICE = "Choice", int, IOKind.JSON
    MULTIPLE_CHOICE = "Multiple choice", int, IOKind.JSON

    # Image types
    IMAGE = "Image", bytes, IOKind.IMAGE
    SEGMENTATION = "Segmentation", bytes, IOKind.IMAGE
    HEAT_MAP = "Heat Map", bytes, IOKind.IMAGE

    # File types
    PDF = "PDF file", bytes, IOKind.FILE
    SQREG = "SQREG file", bytes, IOKind.FILE
    THUMBNAIL_JPG = "Thumbnail jpg", bytes, IOKind.FILE
    THUMBNAIL_PNG = "Thumbnail png", bytes, IOKind.FILE
    OBJ = "OBJ file", bytes, IOKind.FILE
    MP4 = "MP4 file", bytes, IOKind.FILE

    # Legacy support
    CSV = "CSV file", str, IOKind.FILE
    ZIP = "ZIP file", bytes, IOKind.FILE


@dataclass
class Interface:
    slug: str
    relative_path: str
    kind: InterfaceKind

    @property
    def kwarg(self):
        return self.slug.replace("-", "_").lower()

    def load(self):
        if self.kind.io_kind == IOKind.JSON:
            return self._load_json()
        elif self.kind.io_kind == IOKind.IMAGE:
            return self._load_image()
        elif self.kind.io_kind == IOKind.FILE:
            return self._load_file()
        else:
            raise AttributeError(f"Unknown io kind {self.kind.io_kind!r} for {self.kind!r}")

    def save(self, *, data):
        if self.kind.io_kind == IOKind.JSON:
            return self._save_json(data=data)
        elif self.kind.io_kind == IOKind.IMAGE:
            return self._save_image(data=data)
        elif self.kind.io_kind == IOKind.FILE:
            return self._save_file(data=data)
        else:
            raise AttributeError(f"Unknown io kind {self.kind.io_kind!r} for {self.kind!r}")

    def _load_json(self):
        with open(INPUT_PREFIX / self.relative_path, "r") as f:
            return json.loads(f.read())

    def _save_json(self, *, data):
        with open(OUTPUT_PREFIX / self.relative_path, "w") as f:
            f.write(json.dumps(data))

    def _load_image(self):
        input_directory = INPUT_PREFIX / self.relative_path

        mha_files = {f for f in input_directory.glob("*.mha") if f.is_file()}

        if len(mha_files) == 1:
            mha_file = mha_files.pop()
            return SimpleITK.ReadImage(mha_file)
        elif len(mha_files) > 1:
            raise RuntimeError(
                f"More than one mha file was found in {input_directory!r}"
            )
        else:
            raise NotImplementedError

    def _save_image(self, *, data):
        output_directory = OUTPUT_PREFIX / self.relative_path

        output_directory.mkdir(exist_ok=True, parents=True)
        # print (data.GetSpacing())
        file_save_name = output_directory / "overlay.mha"
        # print (file_save_name)

        SimpleITK.WriteImage(data, file_save_name)
        check_file = os.path.isfile(file_save_name)
        # print ("check file", check_file)

    @property
    def _file_mode_suffix(self):
        if self.kind.annotation == str:
            return ""
        elif self.kind.annotation == bytes:
            return "b"
        else:
            raise AttributeError(f"Unknown annotation {self.kind.annotation!r} for {self.kind!r}")

    def _load_file(self):
        with open(INPUT_PREFIX / self.relative_path, f"r{self._file_mode_suffix}") as f:
            return f.read()

    def _save_file(self, *, data):
        with open(OUTPUT_PREFIX / self.relative_path, f"w{self._file_mode_suffix}") as f:
            f.write(data)


INPUT_INTERFACES = [
    Interface(slug="z-score-apparent-diffusion-coefficient-map", relative_path="images/z-score-adc",
              kind=InterfaceKind.IMAGE),
    Interface(slug="skull-stripped-adc", relative_path="images/skull-stripped-adc-brain-mri", kind=InterfaceKind.IMAGE),
]

OUTPUT_INTERFACES = [
    Interface(slug="hypoxic-ischemic-encephalopathy-lesion-segmentation",
              relative_path="images/hie-lesion-segmentation", kind=InterfaceKind.SEGMENTATION),
]

Inputs = make_dataclass(cls_name="Inputs", fields=[(inpt.kwarg, inpt.kind.annotation) for inpt in INPUT_INTERFACES])

Outputs = make_dataclass(cls_name="Outputs",
                         fields=[(output.kwarg, output.kind.annotation) for output in OUTPUT_INTERFACES])


def load() -> Inputs:
    return Inputs(
        **{interface.kwarg: interface.load() for interface in INPUT_INTERFACES}
    )


def get_default_device():
    ######## set device#########
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class SimpNet(nn.Module):
    def __init__(self, output_dim=1, hidden_dim=64, pretrained=True):
        super().__init__()
        from ternausnet.models import UNet16
        self.encoder = UNet16(num_classes=hidden_dim, pretrained=pretrained)
        self.hidden_layer = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.output_layer = nn.Conv2d(hidden_dim, output_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.ReLU()(x)
        x = self.hidden_layer(x)
        x = torch.nn.ReLU()(x)
        return self.output_layer(x)


def aug_predict(model, inp, device):
    # x: (n, 3, d, d)
    prediction = 0
    x = torch.tensor(inp, dtype=torch.float32).to(device)
    prediction += nn.functional.sigmoid(model(x)).detach().cpu().numpy()

    p = nn.functional.sigmoid(model(torch.flip(x, [-1])))
    prediction += torch.flip(p, [-1]).detach().cpu().numpy()

    p = nn.functional.sigmoid(model(torch.flip(x, [-2])))
    prediction += torch.flip(p, [-2]).detach().cpu().numpy()

    p = nn.functional.sigmoid(model(torch.flip(x, [-1, -2])))
    prediction += torch.flip(p, [-1, -2]).detach().cpu().numpy()

    return prediction / 4.


def predict(*, inputs: Inputs) -> Outputs:
    z_adc = inputs.z_score_apparent_diffusion_coefficient_map
    adc_ss = inputs.skull_stripped_adc

    z_adc = SimpleITK.GetArrayFromImage(z_adc).astype(np.float32)
    adc_ss = SimpleITK.GetArrayFromImage(adc_ss).astype(np.float32)
    num, sx, sy = z_adc.shape
    inp = create_input_array(z_adc, adc_ss, 256, channels_first=True)  # (n, 3 256, 256)
    device = get_default_device()
    paths = ["model-a.pt", "model-b.pt", "model-c.pt",
             "model-d.pt", "model-e.pt", "model-f.pt",
             "model-g.pt", "model-h.pt"]

    with torch.no_grad():
        inp = torch.from_numpy(inp).to(device)

        model = SimpNet(pretrained=False)
        model = model.to(device)

        ensemble_prediction = 0
        for path in paths:
            ckpt = torch.load(path, device)
            model.load_state_dict(ckpt["model_state_dict"])
            thresh = ckpt["best_thresh"]
            del ckpt

            out = aug_predict(model, inp, device)
            out = np.squeeze(out, axis=1)
            out = ndimage.zoom(out, (1, sx / 256, sy / 256))
            out = (out > thresh).astype(np.uint8)

            assert out.shape == (num, sx, sy)
            ensemble_prediction = ensemble_prediction + out
            del out

        ensemble_prediction = (ensemble_prediction >= 4).astype(np.uint8)

    hie_segmentation = SimpleITK.GetImageFromArray(ensemble_prediction)

    outputs = Outputs(
        hypoxic_ischemic_encephalopathy_lesion_segmentation=hie_segmentation
    )
    return outputs


def save(*, outputs: Outputs) -> None:
    for interface in OUTPUT_INTERFACES:
        interface.save(data=getattr(outputs, interface.kwarg))


def main() -> int:
    inputs = load()
    outputs = predict(inputs=inputs)
    save(outputs=outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
