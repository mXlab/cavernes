import argparse
from enum import Enum
import torch
from lightweight_gan import Trainer
import copy
import tempfile
import torchvision.transforms as transforms
import os
from pathlib import Path
from tqdm import tqdm


class NoiseType(Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"


class TransitionType(Enum):
    LINEAR = "linear"


def load_model(num=-1, path=None):
    model = Trainer(save_every=1)
    model.load(num, path)
    model.eval()
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--fps", default=30, type=float, help="Frames per second")
parser.add_argument("--duration", default=1, type=float, help="Clip duration (in seconds)")
parser.add_argument("--num-videos", default=1, type=int, help="Number of videos to generate. Each video corresponds to a different latent walk.")
parser.add_argument("--epoch-begin", default=-1, type=int, help="Starting epoch. Negative numbers start from last model")
parser.add_argument("--epoch-end", default=-1, type=int, help="End epoch. Negative numbers start from last model")
parser.add_argument("--model-dir", default=Path("./models/default"), type=Path, help="Folder where the models are saved.")
parser.add_argument("--output-dir", default=Path("./results/default/videos"), type=Path, help="Folder where the videos are saved")
parser.add_argument("--state-begin", default=None, type=str, help='Starting (z, e) state contained in a file exported using the "export-state" option. Overrides "epoch-begin"')
parser.add_argument("--state-end", default=None, type=str, help='End (z, e) state contained in a file exported using the "export-state" option. Overrides "epoch-end" and forces "z-transition" to "linear"')
parser.add_argument("--export-state", default=None, type=str, help='File where the end (z, e) state will be exported')
parser.add_argument("--z-noise-max", default=0.1, type=float, help='Max amount of noise to be added at every step')
parser.add_argument("--z-noise-min", default=0.0, type=float, help='Min amount of noise to be added at every step')
parser.add_argument("--z-noise-steps", default=1, type=int, help='Number of steps (1 corresponds to a single transition)')
parser.add_argument("--z-noise-type", default=NoiseType.UNIFORM, type=NoiseType, help='Type of noise')
parser.add_argument("--z-transition", default=TransitionType.LINEAR, type=TransitionType, nargs='+', help='Type of transition')

parser.add_argument("--batch-size", default=256, type=int, help="Number of videos to process in parallel. Decrease it if you get an out of memory error.")
parser.add_argument("--trunc-psi", default=0.75, type=float, help="Controls the quality/diversity trade-off, the higher the more diverse, the smaller the more quality.")


args = parser.parse_args()

num_frames = int(args.fps*args.duration)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_begin = load_model(args.epoch_begin, args.model_dir)
model_end = load_model(args.epoch_end, args.model_dir)

model = copy.deepcopy(model_begin)

z_begin = model_begin.sample_noise(args.num_videos)
z_end = model_end.sample_noise(args.num_videos)

ratios = torch.linspace(0., 1., num_frames)

with tempfile.TemporaryDirectory() as tmpdirname:
    for j in range(args.num_videos):
        os.makedirs(os.path.join(tmpdirname, str(j)))
        
    for i, ratio in tqdm(enumerate(ratios), total=len(ratios)):
        z = (1-ratio)*z_begin + ratio*z_end
        z = z.to(device)
        model.interpolate(model_begin, model_end, ratio)
        list_img = model.generate(z, args.batch_size, args.trunc_psi)
        for j, img in enumerate(list_img):
            img = transforms.ToPILImage()(img.cpu())
            img.save(os.path.join(tmpdirname, str(j), "%.5i.png"%i))

    os.makedirs(args.output_dir, exist_ok=True)
    for j in range(args.num_videos):
        template_filename = os.path.join(tmpdirname, str(j), "\%05d.png")
        output_path = args.output_dir / ("%i.mp4"%j)
        os.system("ffmpeg -r %i -i %s %s"%(args.fps, template_filename, output_path))

