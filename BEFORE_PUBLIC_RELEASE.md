# Before public release

The repository is structurally ready for GitHub. Complete these evidence-producing steps before presenting it as a fully reproducible release:

1. Run the MNIST `1,7` demo in both `true_only` and `chimeras_only` modes.
2. Commit one comparison image under `assets/demo_results/` and its small CSV/manifest.
3. Run the CIFAR-10 `cat,dog` demo and record leaf-bank test accuracy.
4. Re-run the final CLEVRER, OpenImages, and VidOR configurations from a clean checkout.
5. Replace the transcribed `results/table2.csv` with generated outputs, including seeds and commit hash.
6. Add the public arXiv/paper URL when available; do not upload an anonymous submission marked “do not distribute.”
7. Confirm the preferred repository name, author spelling, contact information, and ORCID in `CITATION.cff`.
8. Set GitHub description and topics, enable Actions, and pin the repository on the profile.
9. Create a `v0.1.0` release only after the demo commands have been executed successfully on a clean environment.
