import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import polars as pl
from pillow_heif import open_heif
from joblib import Parallel, delayed
from functools import partial


def read_image(path, scale=1 / 4):
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


def prep_case(df):
    df = df.with_columns(
        pl.when(pl.col("case").str.contains("1+", literal=True))
        .then(pl.lit("1+"))
        .when(pl.col("case").str.contains("score 0 case 2", literal=True))
        .then(pl.lit("0"))
        .when(pl.col("case").str.contains("3+ D+ 01", literal=True))
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("2+ DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("3+", literal=True))
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("28 Jun HER2 IHC negative", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ D+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ DISH -", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("HER2 0", literal=True))
        .then(pl.lit("0"))
        .when(pl.col("case").str.contains("HER2 score 1", literal=True))
        .then(pl.lit("1+"))
        .when(pl.col("case").str.contains("2+ DISH-", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ Dish -", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ DISH +", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ Dish+", literal=True))
        .then(pl.lit("2+"))
        .when(
            pl.col("case").str.contains(
                "13 Sep HER2 different brightness", literal=True
            )
        )
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("2+DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("HER2 neg case 01", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2 + DISH +", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("score 0", literal=True))
        .then(pl.lit("0"))
        .otherwise(None)
        .alias("ihc_score")
    ).with_columns(
        pl.when(pl.col("ihc_score").is_in(["0", "1+", "2-"]))
        .then(pl.lit(0))
        .otherwise(1)
        .alias("label")
    )
    return df


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
    data = ProgressParallel(n_jobs=workers, n_total_tasks=len(inputs))(
        delayed(partial(func, **kwargs))(input) for input in inputs
    )
    return data


class HENormalizer:
    def fit(self, target):
        pass

    def normalize(self, I, **kwargs):
        raise Exception('Abstract method')

"""
Source code adapted from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class NumpyMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -np.log((I.astype(float)+1)/Io)

        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        #project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:,1:3])

        phi = np.arctan2(That[:,1],That[:,0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1,3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        ''' Normalize staining appearence of H&E stained images

        Example use:
            see test.py

        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        h, w, c = I.shape
        I = I.reshape((-1,3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, maxC[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = np.reshape(Inorm.T, (h, w, c)).astype(np.uint8)


        H, E = None, None

        if stains:
            # unmix hematoxylin and eosin
            H = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
            H[H > 255] = 255
            H = np.reshape(H.T, (h, w, c)).astype(np.uint8)

            E = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
            E[E > 255] = 255
            E = np.reshape(E.T, (h, w, c)).astype(np.uint8)
        return Inorm, H, E
    

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
