import pandas as pd
from fastai.tabular.all import * 
from fastai.learner import save_model, load_model, mk_metric


start_idx = 34010
output_file = 'results/articles_clustered.csv'
learn = load_learner("models/model.pkl")

df_subset = pd.read_csv("data/training/articles_subset.csv")[['article_id','product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no','cluster']] # you can add more features
df = pd.read_csv("data/articles.csv")[['article_id','product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features

clusters = []
subset_length = len(df_subset)
total_length = len(df)
print(subset_length)
df_subset_idx = 0
for index, row in df.iterrows():
    print("Index:",index,"/",total_length)
    # we already have cluster number for it 
    # print(clusters)
    if df_subset_idx < subset_length and str(row['article_id']) == str(df_subset['article_id'][df_subset_idx]):
        clusters.append(int(df_subset['cluster'][df_subset_idx]))
        df_subset_idx += 1
    else:
        #we need to predict
        # print(index)
        row = row.drop(['article_id'])
        row, cluster, prob = learn.predict(row)
        cluster_num = int(cluster)
        clusters.append(cluster_num)

print(clusters)
print(len(clusters))
print(len(df))
df['cluster'] = clusters

df.to_csv(output_file,index=False)



