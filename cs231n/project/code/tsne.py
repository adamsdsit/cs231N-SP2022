import utils
import tsne_utils
from sklearn.model_selection import train_test_split
from openTSNE import TSNE

# load up data
orig_x, orig_y = utils.read_data()
N, H, W, C = orig_x.shape
orig_x = orig_x.reshape(N, H*W*C)
orig_y = utils.classification(orig_y, orig_y)
train_x, test_x, train_y, test_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)

print("%d training samples" % train_x.shape[0])
print("%d test samples" % test_x.shape[0])

tsne = TSNE(
    perplexity=12,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)

embedding_train = tsne.fit(train_x)
tsne_utils.plot(embedding_train, train_y)
print('ok')

embedding_test = embedding_train.transform(test_x)
tsne_utils.plot(embedding_test, test_y)