import time
import numpy as np
import utils
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# load up data
orig_x, orig_y = utils.read_data()
N, H, W, C = orig_x.shape
orig_x = orig_x.reshape(N, H*W*C)
orig_y = utils.classification(orig_y, orig_y)
orig_x = orig_x / 255.

feat_cols = ['pixel'+str(i) for i in range(orig_x.shape[1])]
df = pd.DataFrame(orig_x,columns=feat_cols)
df['y'] = orig_y
df['label'] = df['y'].apply(lambda i:str(i))

df_subset = df.copy()
data_subset = df_subset[feat_cols].values
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls",7),
    data=df_subset,
    legend="full",
    alpha=0.7
)
plt.show()