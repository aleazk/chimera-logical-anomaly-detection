# Logic VAE — experimental research direction

The Chimera evaluator scores supplied rule graphs. This directory explores a complementary generative model over rule syntax and node-level truth probabilities.

`train_vae_probvalue.py`:

- serializes binary rule graphs as postfix programs;
- encodes concept leaves, logical operators, and edge-negation patterns;
- trains a Transformer VAE with token reconstruction and KL objectives;
- predicts node-level probability values from decoder states;
- supports stack-depth-constrained greedy decoding.

The intended longer-term extension is a semantic world model over rule hypotheses, concept/event states, and interventions. It is not part of the reported Chimera anomaly-detection experiments and should be treated as prototype code.
