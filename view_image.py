import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import config as args


def view_image(img_name, img_dim, slice):

    img = np.fromfile(img_name)
    img = np.reshape(img, (img_dim, img_dim))

    plt.imshow(img)
    plt.title(os.path.basename(img_name))
    plt.show()


if __name__ == '__main__':
    patient_id = 21
    info_path = os.path.join(args.datadir, 'Labels', str(args.imdim), 'RFS_all_tumors_zero.csv')
    z_img_path = os.path.join(args.datadir, 'Images/Tumors', str(args.imdim), 'Zero/')

    info = pd.read_csv(info_path)

    pat_rows = info[info['Pat_ID'] == patient_id]
    pat_rows.reset_index(inplace=True)

    for ind in range(len(pat_rows)):
        img_filename = pat_rows.File[ind]
        img_full_path = os.path.join(z_img_path, img_filename)
        print("Slice ", pat_rows.Slice_Num[ind])
        view_image(img_full_path, args.imdim, pat_rows.Slice_Num[ind])


