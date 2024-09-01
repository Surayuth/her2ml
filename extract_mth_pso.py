import cv2
import numpy as np
import histomicstk as htk
import pickle
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob
from functools import partial
from PIL import Image
from pillow_heif import open_heif

class SWARM:
    def __init__(self, gray, D=4, L=0, U=255,
                W_MIN=0.9, W_MAX=0.3, C1=2., C2=2., 
                POP=5, POP_MIN=5, POP_MAX=40, K=3, G=1000, return_hist=False):
        self.gray = gray.reshape(-1)
        I = np.arange(256)
        hist, _ = np.histogram(gray, bins=np.arange(256+1))
        self.cumsum_n = np.cumsum(hist)
        self.cumsum_ip = np.cumsum(I * hist)

        self.D = D
        self.L = L
        self.U = U
        self.W_MIN = W_MIN
        self.W_MAX = W_MAX
        self.C1 = C1
        self.C2 = C2
        self.POP_MIN = POP_MIN
        self.POP_MAX = POP_MAX
        self.K_MAX = K
        self.G = G
        self.V_MAX = 0.2 * (U - L)

        self.w = W_MAX
        self.k = 0
        self.v = np.random.uniform(-1, 1, (POP, D)) * self.V_MAX
        self.x = L + np.random.uniform(size=(POP, D)) * (U - L)
        self.fitness = np.array([self.objective(e) for e in self.x])
        self.best_fitness = self.fitness.max()
        self.p_best = self.x
        self.g_best = self.x[self.fitness.argmax()]
        self.return_hist = return_hist

        self.hist_x = []
        self.hist_fitness = []


    def run(self):
        for g in range(self.G):
            # adaptive inertia
            cond1 = (self.k <= -self.K_MAX)
            cond2 = (self.k >= self.K_MAX)
            pop = len(self.x)
            if cond1:
                if self.w > self.W_MIN:
                    self.w -= 0.1
                    self.k = 0
                # adaptive pop
                if pop >= self.POP_MAX:
                    self.del_pop()
                else:
                    self.add_pop()
            elif cond2:
                if self.w < self.W_MAX:
                    self.w += 0.1
                    self.k = 0
                # adaptive pop
                if pop <= self.POP_MIN:
                    pass
                else:
                    self.del_pop()

            # update p_best / g_best
            new_fitness = np.array([self.objective(e) for e in self.x])
            arg_best = new_fitness.argmax()
            if new_fitness[arg_best] > self.best_fitness:
                self.best_fitness = new_fitness[arg_best]
                self.g_best = self.x[arg_best]
                # update k
                if self.k > 0:
                    self.k += 1
                else:
                    self.k = 1
            else:
                # update k
                if self.k > 0:
                    self.k = -1
                else:
                    self.k -= 1
            is_fitter = new_fitness > self.fitness
            self.p_best[is_fitter] = self.x[is_fitter]
            # update fitness
            self.fitness = new_fitness
            # update v, x
            self.v = self.w * self.v \
                + self.C1 * np.random.uniform() * (self.p_best - self.x) \
                + self.C2 * np.random.uniform() * (self.g_best - self.x)
            self.x = self.x + self.v
            self.x = np.clip(self.x, a_min=self.L, a_max=self.U)
            self.v = np.clip(self.v, a_min=-self.V_MAX, a_max=self.V_MAX)
            
            if self.return_hist:
                self.hist_x.append(self.x.astype(np.uint8))
                self.hist_fitness.append(self.fitness)

    def add_pop(self):
        d = np.random.randint(0, self.D-1)
        lamb = np.random.uniform(size=(1, self.D))
        # init new particle
        np_x = self.g_best.reshape(1, -1) 
        np_v = np.random.uniform(-1, 1, (1, self.D)) * self.V_MAX
        np_p_best = self.L + lamb * (self.U - self.L)
        np_p_best[0, d] = self.g_best[d]
        np_fitness = self.objective(np_x)
        # add corresponding states
        self.p_best = np.append(self.p_best, np_p_best, 0)
        self.v = np.append(self.v, np_v, 0)
        self.x = np.append(self.x, np_x, 0)
        self.fitness = np.append(self.fitness, np_fitness)

    def del_pop(self):
        # random particle to remove
        idx = np.random.randint(0, len(self.x) - 1)
        # remove corresponding states
        self.p_best = np.delete(self.p_best, idx, 0)
        self.v = np.delete(self.v, idx, 0)
        self.x = np.delete(self.x, idx, 0)
        self.fitness = np.delete(self.fitness, idx, 0)
        
    def objective(self, ths):
        ths = np.sort(np.append(ths, 256)).astype(np.uint8)
        w = np.zeros(len(ths) + 1)
        u = np.zeros(len(ths) + 1)

        curr_cumsum_n = 0
        curr_cumsum_ip = 0
        for i, th in enumerate(ths-1):
            cum_n = self.cumsum_n[th]
            cum_ip = self.cumsum_ip[th]
            w[i] = cum_n - curr_cumsum_n
            u[i] = (cum_ip - curr_cumsum_ip)
            curr_cumsum_n = cum_n
            curr_cumsum_ip = cum_ip
        ut = (u / w.sum()).sum()
        return (w / w.sum() * (u / (w + 1e-8) - ut) ** 2).sum()
    
    def get_best(self):
        return np.sort(self.g_best.astype(np.uint8)).tolist()

    def get_var(self):
        return self.objective(self.g_best)
    
    def get_hist(self):
        return self.hist_x, self.hist_fitness

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

def get_dab(imInput):
    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
            'dab',        # cytoplasm stain
            ]         # set to null if input contains only two stains

    # create stain matrix
    W_init = np.array([stain_color_map[st] for st in stains]).T

    # Compute stain matrix adaptively
    sparsity_factor = 0.5

    I_0 = 255
    im_sda = htk.preprocessing.color_conversion.rgb_to_sda(imInput, I_0)
    W_est = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(
        im_sda, W_init, sparsity_factor,
    )

    # perform sparse color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
        imInput,
        htk.preprocessing.color_deconvolution.complement_stain_matrix(W_est),
        I_0,
    )
    gray = imDeconvolved.Stains[:,:,1]
    return gray

def get_q_img(gray, mth):
    q_img = np.zeros_like(gray)
    th1 = 0
    img_list = []
    for i, th in enumerate(mth + [256]):
        th2 = th
        mask = (th1 <= gray) & (gray < th2)
        img_list.append(mask.T)
        q_img += (i * mask).astype(np.uint8)
        th1 = th
    return np.array(img_list[:3]).T * 255

def get_result(path, dst, level, resize_scale):
    # create a new folder from the path
    path = Path(path)
    dst_dir = dst / path.parent / path.stem
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    img = read_image(str(path), resize_scale)
    gray = get_dab(img)
    swarm = SWARM(gray, D=level)
    swarm.run()
    mth = swarm.get_best()
    q_img = get_q_img(gray, mth)
    otsu_img = 255 - cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cv2.imwrite(str(dst_dir / "q_img.png"), cv2.cvtColor(q_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dst_dir / "otsu_img.png"), otsu_img)
    with open(dst_dir / "mth.pkl", 'wb') as f:
        pickle.dump(mth, f)
    return True

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

if __name__ == "__main__":
    level = 4
    resize_scale = 1/2
    src = "Data_Chula"
    dst = Path(f"extracted_data_pso_{level}_{resize_scale}")
    workers=8

    if not dst.is_dir():
        dst.mkdir(parents=True)
    paths = glob(f"{src}/*/*")
    # exts = [path.split(".")[-1] for path in paths] 
    ProgressParallel(n_jobs=workers, n_total_tasks=len(paths))(delayed(partial(get_result, dst=dst, level=level, resize_scale=resize_scale))(path) for path in paths)