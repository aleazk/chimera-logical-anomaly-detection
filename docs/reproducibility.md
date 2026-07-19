# Reproducibility checklist

For each reported experiment, preserve:

- the repository commit hash;
- Python, PyTorch, DGL, CUDA, and driver versions;
- the full command line or config file;
- random seeds;
- concept vocabulary and rule manifest;
- dataset split and filtering decisions;
- leaf-bank metrics;
- per-rule scores before aggregation;
- generated checkpoints and cache lineage metadata.

The current `results/table2.csv` is a transcription of the manuscript table. It should be replaced by automatically exported results after the final configurations are frozen.
