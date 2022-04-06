from email.policy import strict
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


#use subdir 26
subdir = "026"

base_dir = "results/data/"

#images
image_base_dir = "data/images"
images_dir= os.path.join(image_base_dir, subdir)
file_name = os.path.join(base_dir, "cluster_results_" + subdir + ".csv")
df = pd.read_csv (file_name,header=None)

output_path = "results/visuals/" + subdir

print(df)


my_dict = {}

if not os.path.exists(output_path):
    os.makedirs(output_path)

for index, row in df.iterrows():
    # print(row)
    image_name= "0" + str(row[0]) + ".jpg"
    cluster = str(int(row[1]))

    image_path = os.path.join(images_dir, image_name)

    if cluster not in my_dict:
        my_dict[cluster] = []

    my_dict[cluster].append(image_path)

print(my_dict)


for cluster,image_list in my_dict.items():

    print("Cluster: ", cluster)
    # f , axarr = plt.subplots(len(image_list),1,figsize=(20, 20))

    # plt.subplots_adjust(wspace=0, hspace=0)

    print(len(image_list))
    print(image_list)

    images = []
    first = None
    for i,img in enumerate(image_list):
        pic = Image.open(img).convert('RGB')
        if first is None:
            first = pic
        images.append(pic)

        # axarr[i].imshow(pic)
        # axarr[i].set_xticklabels([])
        # axarr[i].set_yticklabels([])
        # axarr[i].set_aspect('equal')

    # print(cluster)
    file_output_name = "cluster_"+str(int(cluster))+"_subdir_"+subdir +".pdf"

    first.save(os.path.join(output_path,file_output_name), save_all=True, append_images=images[1:])
    # print(file_output_name)s

    # plt.suptitle("Cluster " + str(int(cluster)))
    # plt.savefig(os.path.join(output_path, file_output_name))
    

    



