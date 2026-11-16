"""VESPA for MountainCarContinuous (stock gymnax dynamics).

Zonotope verification of SYMPOL decision-tree policies against the goal (reachability) property,
counterexample regions from the violating zonotopes, and PPO fine-tuning with mixed starts.

Entry points: ``python -m vespa.mountaincar.run`` (one input policy x method x fine-tuning seed) and
``python -m vespa.mountaincar.verify`` (verify one policy).

Importing the package enables float64 in JAX, which the verifier and the counterexample walk need for
their outward rounding.
"""
import jax

jax.config.update("jax_enable_x64", True)
