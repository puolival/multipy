# MultiPy
Testing multiple hypotheses simultaneously increases the number of false 
positive findings if the corresponding p-values are not corrected. While this 
multiple testing problem is well known, the classic and advanced correction 
methods are yet to be implemented into a coherent Python package. This package 
sets out to fill this gap by implementing methods for controlling the 
family-wise error rate (FWER) and the false discovery rate (FDR).

## Learn by trying it yourself

Download the <a href="https://github.com/puolival/multipy/archive/master.zip">
program code</a>, install <a href="https://jupyter.org/">Jupyter</a> and do 
our interactive exercises! You can use the supplied data or your own. A preview 
of the first exercise is available 
<a href="https://github.com/puolival/multipy/blob/master/exercise1.ipynb">here</a>.

## Dependencies

The required packages are 
<a href="http://www.numpy.org/">NumPy</a>,
<a href="https://www.scipy.org/">SciPy</a>,
<a href="https://matplotlib.org/">Matplotlib</a>, and
<a href="https://seaborn.pydata.org">Seaborn</a>.

## Quick example

```python
from adaptive import lsu
from data import neuhaus
import matplotlib.pyplot as plt
from viz import plot_pval_hist

pvals = neuhaus()
significant_pvals = lsu(pvals, q=0.05)
print zip(['{:.4f}'.format(p) for p in pvals], significant_pvals)
fig = plot_pval_hist(pvals)
plt.show() # show figure
```

```python
[('0.0001',  True), ('0.0004',  True), ('0.0019',  True), 
 ('0.0095',  True), ('0.0201', False), ('0.0278', False), 
 ('0.0298', False), ('0.0344', False), ('0.0459', False), 
 ('0.3240', False), ('0.4262', False), ('0.5719', False), 
 ('0.6528', False), ('0.7590', False), ('1.0000', False)]
```
<img src="https://puolival.github.io/multipy/figs/neuhaus.png" alt="neuhaus.png" />
