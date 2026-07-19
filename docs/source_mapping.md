# Source mapping

| Public path | Source material |
|---|---|
| `src/chimera_logic/evaluator.py` | original `evaluator.py`, packaged with exact semantics imported from `semantics.py` |
| `src/chimera_logic/trainer.py` | original `trainer.py`, packaged and with consistent edge-negation feature encoding |
| `demos/mnist_forbidden_conjunction.py` | original `mnist_and_pair_grids.py`, focused default `1,7` |
| `demos/cifar10_forbidden_conjunction.py` | original `cifar10_and_pair_grids.py`, focused default `cat,dog` |
| `experiments/clevrer/train.py` | original CLEVR/CLEVRER experiment driver |
| `experiments/openimages/train.py` | original OpenImages experiment driver |
| `experiments/vidor/train.py` | original VidOR experiment driver |
| `research/logic_vae/train_vae_probvalue.py` | original Logic VAE prototype |

The public package keeps the scientific code close to the research originals while providing clearer entry points, metadata, tests, and repository organization.
