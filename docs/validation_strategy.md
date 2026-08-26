# Validation Strategy

## A. Random split

Purpose:

- Smoke-test the training and inference code path.
- Catch obvious bugs in loader, batching, and metric wiring.

Why it is not enough:

- It leaks context and perturbation information across train and validation.
- It is usually over-optimistic for this challenge.

## B. Perturbation holdout

Purpose:

- Hold out a subset of perturbations entirely.
- Test whether the model can extrapolate to unseen perturbations within seen contexts.

Why it matters:

- It is closer to the challenge than random splitting.
- It tests perturbation generalization, but still not the hardest challenge axis.

## C. Cell-context holdout

Purpose:

- Train on one or more contexts.
- Validate on a fully unseen context.

Why it matters:

- This is the closest local analogue to the 2026 final test.
- It tests context generalization, which is the core difficulty of the challenge.

Recommended ordering:

1. Start with random split for pipeline sanity.
2. Add perturbation holdout to test unseen-target generalization.
3. Use cell-context holdout as the primary selection criterion.

