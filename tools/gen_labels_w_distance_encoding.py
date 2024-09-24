from pathlib import Path
import sys

import numpy as np
import nibabel as nib

from brainsynth.constants import IMAGE
gl = IMAGE.generation_labels

def distance_to_label(d, max_d, n, start_label):
    # f = len(d)/2/max_d
    return np.round((d+max_d)*n/max_d + start_label)

def label_to_distance(label, max_d, n, start_label):
    # f = len(label)/2/max_d
    return (label-start_label)/n*max_d - max_d

def distance_to_pv(x, i=1.0):
    """Label to PV fraction."""
    # sigmoid
    return 1/(1 + np.exp(-i * x))

def pv_to_label(f, start, end):
    return np.round(f * (end-start) + start)


# start_index = 200
# n = 50
# max_dist = 3.0
# f = (n-1)/2/max_dist
# d = np.linspace(-max_dist,max_dist,n)
# x = distance_to_label(d, max_dist, (n-1)/2, start_index)
# y = label_to_distance(x, max_dist, (n-1)/2, start_index)

# print(d)
# print(y)
# print(np.abs(d-y).mean())
# print(x)

def dist_map_to_dist(x):
    return (x - 128) / 20

if __name__ == "__main__":
    index = int(sys.argv[1])

    max_dist = 3.0

    d = Path("/mnt/projects/CORTECH/nobackup/training_data/")
    outd = Path("/mnt/scratch/personal/jesperdn/new_gen_labels")

    filename = sorted(d.glob("*generation_labels.nii"))[index]
    ds, subject = filename.stem.split(".")[:2]

    gen = nib.load(d / f"{ds}.{subject}.generation_labels.nii")

    img_lw = nib.load(d / f"{ds}.{subject}.lw_dist_map.nii")
    img_rw = nib.load(d / f"{ds}.{subject}.rw_dist_map.nii")
    img_lp = nib.load(d / f"{ds}.{subject}.lp_dist_map.nii")
    img_rp = nib.load(d / f"{ds}.{subject}.rp_dist_map.nii")

    # choose lowest distance
    data = np.stack(
        (dist_map_to_dist(img_lw.get_fdata()).ravel(),
        dist_map_to_dist(img_rw.get_fdata()).ravel(),
        dist_map_to_dist(img_lp.get_fdata()).ravel(),
        dist_map_to_dist(img_rp.get_fdata()).ravel()
        )
    )
    abs_data = np.abs(data)
    idx = abs_data.argmin(0)
    valid_data = np.where(
        abs_data[idx, np.arange(len(idx))] <= max_dist
    )[0]

    gen_data = gen.get_fdata()
    shape = gen_data.shape
    gen_data = gen_data.ravel()
    pv_mask = np.isin(gen_data.astype(int), (gl.white, gl.gray, gl.csf)) | (gen_data >= gl.pv.white) & (gen_data <= gl.pv.csf)

    # (white, pial)
    subsets = [ [(0,1), (gl.pv.white, gl.pv.gray)], [(2,3), (gl.pv.gray, gl.pv.csf)] ]

    for subset in subsets:
        ind0 = subset[0]
        pvs = subset[1]
        half_size = (pvs[1]-pvs[0])/2 # halfway point

        sel = np.where(np.isin(idx[valid_data], subset))[0]

        valid_sel = valid_data[sel]
        valid_sel = valid_sel[pv_mask[valid_sel]]
        data_sel = data[idx[valid_sel], valid_sel]

        labels = distance_to_label(data_sel, max_dist, half_size, pvs[0])

        gen_data[valid_sel] = labels.astype(gen_data.dtype)

    gen_data = gen_data.reshape(shape)
    # new generation_labels
    img = nib.Nifti1Image(gen_data.astype(gen.get_data_dtype()), gen.affine)
    # img.to_filename("/home/jesperdn/nobackup/testgen.nii")
    img.to_filename(outd / f"{ds}.{subject}.generation_labels_dist.nii")


# img = nib.load("/mnt/scratch/personal/jesperdn/ABIDE.sub-0001.generation_labels_dist.nii")
# data = img.get_fdata()

# def transform_gen_labels_dist(data, max_dist=3.0):
#     # [1.0, 5.0]
#     rho = np.random.rand() * 4 + 1.0

#     info = [(gl.pv.white, gl.pv.gray), (gl.pv.gray, gl.pv.csf)]
#     for pv in info:

#         half_size = (pv[1]-pv[0])/2 # halfway point
#         mask = (data >= pv[0]) & (data <= pv[1])
#         labels = data[mask]

#         new_labels = pv_to_label(
#             distance_to_pv(
#                 label_to_distance(labels, max_dist, half_size, pv[0]),
#                 rho,
#             ),
#             *pv,
#         )

#         data[mask] = new_labels

#     return data

# out = nib.Nifti1Image(data.astype(img.get_data_dtype()), img.affine)
# out.to_filename(f"/mnt/scratch/personal/jesperdn/ABIDE.sub-0001.generation_labels_dist1.nii")
