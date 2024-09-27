import numpy as np
import polars as pl
from skimage import measure

def init_hist(bin_img):
    life_dict = {
    }
    labeled_img = measure.label(bin_img)
    unique_labels = np.unique(labeled_img[labeled_img > 0])
    for label in unique_labels:
        life_dict[int(label)] = {
            "birth": 0,
            "death": None
        }
    if len(unique_labels) > 0:
        max_cur_label = np.max(unique_labels)
    else:
        max_cur_label = 0
    return life_dict, labeled_img, max_cur_label

def update_hist(prev_labeled_img, cur_bin, life_dict, max_cur_label, timestep):
    labeled_img = measure.label(cur_bin)
    ids = np.unique(labeled_img)[1:] # exclude blackground
    new_labeled_img = np.zeros(labeled_img.shape).astype(int)

    if np.sum(cur_bin) > 0:
        # find overlap objects
        overlap_hist = []
        new_non_overlap_ids = []
        for id in ids:
            overlap_ids = (labeled_img == id) * prev_labeled_img
            if np.max(overlap_ids) > 0:
                overlap_id = np.max(overlap_ids[overlap_ids > 0])
                overlap_hist.append(
                    [id, overlap_id] # new_id, old_id
                )
            else:
                new_non_overlap_ids.append(id)
        
        if len(overlap_hist) > 0:
            df = pl.DataFrame(np.array(overlap_hist), schema=["new_id", "old_id"]) \
                .with_columns(
                    pl.len().over("old_id").alias("intersect")
                ) \
                .with_columns(
                    pl.when(pl.col("intersect") > 1)
                    .then(0).otherwise(1)
                    .alias("alive")
                )

            death_df = df.filter(pl.col("alive") == 0)
            alive_df = df.filter(pl.col("alive") == 1)
            alive_prev_ids = alive_df.select("old_id").unique().to_series().to_list() # id of prev
            new_cur_ids = sorted(death_df.select("new_id").unique().to_series().to_list()) # id of cur
            #print(alive_prev_ids)

            for i, new_id in enumerate(new_cur_ids):
                new_labeled_img += (labeled_img == new_id) * (max_cur_label + i + 1)
            for alive_id in alive_prev_ids:
                new_labeled_img += ((prev_labeled_img == alive_id) & (labeled_img > 0)) * alive_id

            # update death -> life_dict
            for id in life_dict.keys():
                if (id not in new_labeled_img) and (life_dict[id]["death"] is None):
                    life_dict[id]["death"] = timestep

            # update birth -> life_dict
            for i in range(len(new_cur_ids)):
                new_label = int(max_cur_label + i + 1)
                life_dict[new_label] = {"birth": timestep, "death": None}
            max_cur_label += len(new_cur_ids)
        
        if len(new_non_overlap_ids) > 0:
            for i, new_id in enumerate(new_non_overlap_ids):
                new_labeled_img += (labeled_img == new_id) * (max_cur_label + i + 1)
            # update birth -> life_dict
            for i in range(len(new_non_overlap_ids)):
                new_label = int(max_cur_label + i + 1)
                life_dict[new_label] = {"birth": timestep, "death": None}
            max_cur_label += len(new_non_overlap_ids)
    else:
        # all dead
        for k in life_dict.keys():
            if life_dict[k]["death"] is None:
                life_dict[k]["death"] = timestep
        
    return life_dict, new_labeled_img, max_cur_label

def prep_mask(mask):
    labeled_image = measure.label(mask)
    regions = measure.regionprops(labeled_image)
    h, w = labeled_image.shape
    min_area = h * w * (0.01) ** 2
    max_area = h * w * (0.05) ** 2
    selected_labels = []
    for r in regions:
        area = r.area
        if (area > min_area) and (area < max_area):
            selected_labels.append(r.label)
    # Create a boolean mask where elements in arr are in values_to_select
    mask = np.isin(labeled_image, selected_labels)
    return mask