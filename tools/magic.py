from tools.utils import Utils
from tools.models import D2KeypointModel
import os
import argparse
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description="AIStudio-KeypointDetection")
    parser.add_argument("--config_file", required=True, help="model configs")
    parser.add_argument("--image_path", type=str, required=True, help="input image path")
    parser.add_argument("--result_path", type=str, default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.result_path:
        args.result_path = str(Path(args.image_path).parent / 'results')
    Path(args.result_path).mkdir(exist_ok=True, parents=True)
    utils = Utils()
    model = D2KeypointModel(args)

    image = utils.load_image(args.image_path)
    _, pred_boxes = model(image)

    results = {}
    results['images'] = {
        'image_name': args.image_path,
        'image_shape': image.shape
    }
    results['bounding_boxes'] = [[int(i) for i in b] for b in pred_boxes]

    image_name = os.path.basename(args.image_path)
    json_fp = os.path.join(args.result_path, f"result_{Path(image_name).stem}.json")
    with open(json_fp, 'w') as w_f:
        json.dump(results, w_f, indent=4)

    image = utils.draw_rectangle(image, pred_boxes, thickness=2)

    utils.save_images(image, args.result_path, image_name)

    return image


if __name__ == '__main__':
    image = main()
