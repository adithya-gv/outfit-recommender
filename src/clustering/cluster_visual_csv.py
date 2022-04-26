from email.policy import strict
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description="Visual Clustering")
parser.add_argument(
    "--c_type",
    type=str,
    default='agg',
    help="name of clustering alg csv",
)
parser.add_argument(
    "--subdir",
    type=str,
    default='036',
    help="name of clustering alg csv",
)
args = parser.parse_args()

#use subdir 26
subdir = args.subdir

base_dir = "results/data/"

#images
image_base_dir = "data/images"
images_dir= os.path.join(image_base_dir, subdir)
if args.c_type == "agg":
    file_name = os.path.join(base_dir, "cluster_results_agg_" + subdir + ".csv")
else:
    file_name = os.path.join(base_dir, "cluster_results_" + subdir + ".csv")
df = pd.read_csv (file_name,header=None)



if args.c_type == "agg":     
    output_path = "results/visuals/agg/" + subdir
else:
    output_path = "results/visuals/kmeans/" + subdir


print("dataframe", df)


my_dict = {}

if not os.path.exists(output_path):
    os.makedirs(output_path)

for index, row in df.iterrows():
    image_name= "0" + str(row[0]) + ".jpg"
    cluster = str(int(row[1]))

    image_path = os.path.join(images_dir, image_name)

    if cluster not in my_dict:
        my_dict[cluster] = []

    my_dict[cluster].append(image_path)

print(my_dict)

lastrow = 0
lastCol = 0
for cluster,image_list in my_dict.items():

    print("Cluster: ", cluster)

    # 5 images per row
    # rows = round(len(images) / 5)
    num_rows = int(len(image_list) / 5 + 1)
    one_row = False
    if num_rows == 1:
        one_row = True
    f , axarr = plt.subplots(num_rows,5,figsize=(17,20))
    plt.subplots_adjust(wspace=0, hspace=0)

    print("Image list", image_list)

    # images = []
    # first = None

    for i,img in enumerate(image_list):
        pic = Image.open(img).convert('RGB')
        row = int(i / 5)
        col = int(i % 5)


        if one_row:
            axarr[col].imshow(pic)
            axarr[col].set_xticklabels([])
            axarr[col].set_yticklabels([])
            axarr[col].set_aspect('equal')
        else:
            axarr[row,col].imshow(pic)
            axarr[row,col].set_xticklabels([])
            axarr[row,col].set_yticklabels([])
            axarr[row,col].set_aspect('equal')
        last_row = row
        last_col = col

    # removes empty plots in the image
    if (lastrow == num_rows - 1):
        if one_row:
            for i in range(last_col + 1, 5):
                axarr[i].set_axis_off()
        else:
            for i in range(last_col + 1, 5):
                axarr[lastrow,i].set_axis_off()
    else:
        for i in range(last_col + 1, 5):
            axarr[lastrow,i].set_axis_off()
        #will only ever have 1 extra row than needed
        for i in range(0, 5):
            axarr[num_rows - 1,i].set_axis_off()
        

    # print(cluster)
    file_output_name = "cluster_"+str(int(cluster))+"_subdir_"+subdir +".pdf"

    # first.save(os.path.join(output_path,file_output_name), save_all=True, append_images=images[1:])
    # print(file_output_name)

    plt.suptitle("Cluster " + str(int(cluster)))
    plt.savefig(os.path.join(output_path, file_output_name))
    

    



