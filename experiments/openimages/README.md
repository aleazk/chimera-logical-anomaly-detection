# Openimages experiment

This directory contains the preserved research driver for the Openimages experiment. It is not a lightweight demo.

Install the package from the repository root before running:

```bash
pip install -e ".[all]"
python experiments/openimages/train.py --help
```

The dataset must be obtained separately and arranged according to the path assumptions documented at the top of `train.py`. Run outputs, checkpoints, and caches should remain outside version control.
