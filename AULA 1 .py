import sompy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Geração de dados de exemplo
data, true_labels = make_blobs(n_samples=300, centers=4, random_state=42)

# Normalização dos dados
data = data / data.max(axis=0)

# Definição dos parâmetros da SOM
map_size = (10, 10)
som = sompy.SOMFactory.build(data, map_size, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
som.train(n_job=1, verbose='info')

# Visualização dos resultados
v = sompy.mapview.View2DPacked(10, 10, 'Test', text_size=8)  
v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6) 
plt.show()
