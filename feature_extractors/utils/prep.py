import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pillow_heif import open_heif
from joblib import Parallel, delayed
from functools import partial

def read_image(path, scale=1/4):
    try:
        if Path(path).suffix.lower() == ".heic":
            img = open_heif(path, convert_hdr_to_8bit=True)
        else:
            img = Image.open(path)
        rgb = np.asarray(img)
        H, W = rgb.shape[:-1]
        new_H = int(H * scale)
        new_W = int(W * scale)
    except:
        print(path)
        raise
    return cv2.resize(rgb, (new_W, new_H))

class ProgressParallel(Parallel):
    def __init__(self, n_total_tasks=None, **kwargs):
        super().__init__(**kwargs)
        self.n_total_tasks = n_total_tasks

    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.n_total_tasks:
            self._pbar.total = self.n_total_tasks
        else:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def mod_parallel(func, workers, inputs, **kwargs):
    data = ProgressParallel(
            n_jobs=workers, 
            n_total_tasks=len(inputs)
        )(
            delayed(
                partial(
                    func,
                    **kwargs
                )
            )(input) 
            for input in inputs
        )
    return data