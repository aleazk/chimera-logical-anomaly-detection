# Implementation notes

## Edge-negation features

The public package uses a single convention for neural gate inputs:

- `0.0`: non-negated edge;
- `1.0`: negated edge.

This convention is used consistently in ordinary propagation, Chimera training, frozen-gate propagation, and the standalone Chimera utilities.

## Scientific status

The repository is an alpha research implementation. The large experiment drivers preserve the working research scripts to maximize traceability. The core library and demos are the preferred entry points for new users.
