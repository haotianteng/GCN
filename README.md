# GCN is an extended package for reconstructing causal connection by using Bayesian Network, it's built upon **networkX** and **pgmpy**.

# INSTALLATION
Clone the repository and add GCN folder into PYTHONPATH.  
```bash
git clone https://github.com/haotianteng/GCN.git
cd GCN
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/:$PYTHONPATH"
```

# USAGE
```python
import numpy as np
from gcn import GaussianCausalNetwork
from networkx.generators.random_graphs import random_regular_graph
N_NODE = 5
G = random_regular_graph(d = 4, n = N_NODE)
weights = np.random.rand(N_NODE,N_NODE+2)
bn = GaussianCausalNetwork(G,weight_matrix = weights)
```


