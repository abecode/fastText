preprocess data

```
cat <(zcat /data/headlines.gz| cut -f 3) <(zcat /data/headlines_sn.gz| cut -f 4) | perl wikifil.mod.pl > data.txt
```

run fasttext skipgram

```
./fasttext skipgram -input data.txt -output model
```


in ipython

```
import numpy as np
vectors = {}
with open("model.vec") as f:
  for line in f:
    tab = line.split()
    if len(tab) == 2:
      continue
    word = tab[0]
    vec = np.array(tab[1:], dtype=float)
    if not word in vectors:
	  vectors[word] = vec

W =  np.empty((len(vectors),100)) # 100 is number of neuron units
for i,w in enumerate(vectors):
  W[i,:] = vectors[w]
words = [w for w in vectors]
W[words.index("drop"),:] # find word vector for drip
WW = W*W
# similarity = A*B / sqrt(A**2) / sqrt(B**2)
# but since the query is the same/constant, I ignore it
res = (W[words.index("drop"),:] * W).sum(1) / np.sqrt(WW.sum(1))
[words[x] for x in np.argsort(-res)[0:100]]

```
