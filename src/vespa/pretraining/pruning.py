"""Pruning of a SYMPOL checkpoint (paper Sec. on input policies).

Two reductions that never change the action of the greedy tree on any observation inside the observation space:

1. A split whose threshold lies outside the observation bounds of its feature, or outside the interval that the
   splits above it already impose on that feature, always sends every state to the same child. The node is replaced
   by that child.
2. For a discrete action space, two sibling leaves with the same greedy action (argmax) are merged.

The tree arrays keep their shape: only ``leaf_array`` is rewritten, every leaf of a collapsed subtree receives the
values of the leaf that replaces it, and the split parameters stay untouched. The result is written to
``<checkpoint>-pruned`` in the same checkpoint format.

    python -m vespa.pretraining.pruning <checkpoint>
"""
import argparse
import os

import numpy as np
import orbax.checkpoint

from vespa.pretraining.checkpoint import restore_raw
from vespa.pretraining.envs import OBSERVATION_BOUNDS


def prune_and_merge_tree(node, split_ranges, constraints=None, continuous=False):
    """Prune a dict tree (see ``array_to_dict_tree``) and return its replacement.

    ``split_ranges`` maps a feature to its observation bounds ``(low, high)``. ``constraints`` maps a feature to the
    pair ``(smallest threshold of a left turn above, largest threshold of a right turn above)``.
    """
    if constraints is None:
        constraints = {}
    if node['type'] == 'leaf':
        return node

    split_index = node['split_index']
    split_value = node['split_value']

    def collapse_vacuous(reachable_key):
        # The node keeps its position in the array tree (its split parameters stay in place), so only a leaf child
        # may take its place: a leaf is written to every descendant slot, which is independent of the position.
        child = prune_and_merge_tree(node[reachable_key], split_ranges, constraints, continuous=continuous)
        if child['type'] == 'leaf':
            return child
        node[reachable_key] = child
        return node

    if split_index in split_ranges:
        min_value, max_value = split_ranges[split_index]
        if split_value < min_value or split_value > max_value:
            if split_value < min_value:
                return collapse_vacuous('right_child')
            else:
                return collapse_vacuous('left_child')

    if split_index in constraints:
        min_constraint, max_constraint = constraints[split_index]
        if (split_value >= min_constraint) or (split_value <= max_constraint):
            if split_value <= max_constraint:
                return collapse_vacuous('right_child')
            else:
                return collapse_vacuous('left_child')

    new_constraints_left = constraints.copy()
    new_constraints_right = constraints.copy()
    if split_index in new_constraints_left:
        new_constraints_left[split_index] = (min(split_value, new_constraints_left[split_index][0]),
                                             new_constraints_left[split_index][1])
    else:
        new_constraints_left[split_index] = (split_value, -np.inf)
    if split_index in new_constraints_right:
        new_constraints_right[split_index] = (new_constraints_right[split_index][0],
                                              max(split_value, new_constraints_right[split_index][1]))
    else:
        new_constraints_right[split_index] = (np.inf, split_value)

    node['left_child'] = prune_and_merge_tree(node['left_child'], split_ranges, new_constraints_left,
                                              continuous=continuous)
    node['right_child'] = prune_and_merge_tree(node['right_child'], split_ranges, new_constraints_right,
                                               continuous=continuous)

    if (not continuous and
            node['left_child'] and node['left_child']['type'] == 'leaf' and
            node['right_child'] and node['right_child']['type'] == 'leaf' and
            np.argmax(node['left_child']['distribution']) == np.argmax(node['right_child']['distribution'])):
        return node['left_child']
    return node


def array_to_dict_tree(e, node_id, depth, max_depth, params_np, indices_np):
    """Convert tree ``e`` of the parameter arrays into a nested dict tree (root: ``node_id=0``, ``depth=0``).

    Internal nodes carry the selected observation feature (``split_index``) and its threshold (``split_value``).
    """
    if depth == max_depth:
        leaf_id = node_id - (2 ** max_depth - 1)
        return {'type': 'leaf', 'distribution': params_np['leaf_array'][e][leaf_id].tolist()}
    node_sel_idx = int(np.argmax(params_np['split_idx_array'][e][node_id]))
    obs_feat = int(indices_np['features_by_estimator'][e][node_sel_idx])
    threshold = float(params_np['split_values'][e][node_id][node_sel_idx])
    return {
        'type': 'internal',
        'split_index': obs_feat,
        'split_value': threshold,
        'left_child': array_to_dict_tree(e, 2 * node_id + 1, depth + 1, max_depth, params_np, indices_np),
        'right_child': array_to_dict_tree(e, 2 * node_id + 2, depth + 1, max_depth, params_np, indices_np),
    }


def fill_all_leaves(node_id, depth, max_depth, distribution, leaf_array_out, e):
    """Write one leaf value to every leaf below ``node_id``."""
    if depth == max_depth:
        leaf_id = node_id - (2 ** max_depth - 1)
        leaf_array_out[e][leaf_id] = np.array(distribution, dtype=leaf_array_out.dtype)
        return
    fill_all_leaves(2 * node_id + 1, depth + 1, max_depth, distribution, leaf_array_out, e)
    fill_all_leaves(2 * node_id + 2, depth + 1, max_depth, distribution, leaf_array_out, e)


def fill_leaf_array(node, node_id, depth, max_depth, leaf_array_out, e):
    """Write a pruned dict tree back into ``leaf_array_out[e]``."""
    if depth == max_depth:
        leaf_id = node_id - (2 ** max_depth - 1)
        leaf_array_out[e][leaf_id] = np.array(node['distribution'], dtype=leaf_array_out.dtype)
        return
    if node['type'] == 'leaf':
        fill_all_leaves(node_id, depth, max_depth, node['distribution'], leaf_array_out, e)
        return
    fill_leaf_array(node['left_child'], 2 * node_id + 1, depth + 1, max_depth, leaf_array_out, e)
    fill_leaf_array(node['right_child'], 2 * node_id + 2, depth + 1, max_depth, leaf_array_out, e)


def count_leaves(node):
    """Number of leaves of a dict tree."""
    if node['type'] == 'leaf':
        return 1
    return count_leaves(node['left_child']) + count_leaves(node['right_child'])


def prune_checkpoint(checkpoint_path):
    """Prune every tree of a checkpoint and save ``<checkpoint>-pruned``.

    Returns the output path and the number of leaves of each pruned tree.
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = checkpoint_path.rstrip('/') + '-pruned'
    raw = restore_raw(checkpoint_path)
    env_id = str(raw['env_id'])
    low, high = OBSERVATION_BOUNDS[env_id]
    split_ranges = {i: (float(np.float64(low[i])), float(np.float64(high[i]))) for i in range(len(low))}
    continuous = str(raw['action_type']) == 'continuous'

    params = raw['actor']['params']
    indices = raw['actor']['indices']
    leaf_arr = np.asarray(params['leaf_array'])  # (n_estimators, 2^depth, action_dim)
    n_estimators = leaf_arr.shape[0]
    depth = int(np.log2(leaf_arr.shape[1]))
    params_np = {
        'split_values': np.asarray(params['split_values']),
        'split_idx_array': np.asarray(params['split_idx_array']),
        'leaf_array': leaf_arr,
    }
    indices_np = {'features_by_estimator': np.asarray(indices['features_by_estimator'])}

    new_leaf_array = leaf_arr.copy()
    leaves = []
    for e in range(n_estimators):
        pruned = prune_and_merge_tree(array_to_dict_tree(e, 0, 0, depth, params_np, indices_np), split_ranges,
                                      continuous=continuous)
        fill_leaf_array(pruned, 0, 0, depth, new_leaf_array, e)
        leaves.append(count_leaves(pruned))
        print(f"[pruning] {env_id} tree {e}: {2 ** depth} -> {leaves[-1]} leaves")

    new_params = dict(params)
    new_params['leaf_array'] = new_leaf_array
    new_actor = {'params': new_params, 'indices': indices}
    new_checkpoint = dict(raw)
    new_checkpoint['actor'] = new_actor
    new_checkpoint['sympol'] = new_actor
    orbax.checkpoint.PyTreeCheckpointer().save(output_dir, new_checkpoint, force=True)
    print(f"[pruning] saved {output_dir}")
    return output_dir, leaves


def main():
    parser = argparse.ArgumentParser(description="Prune a SYMPOL checkpoint and save <checkpoint>-pruned.")
    parser.add_argument('checkpoint', help="checkpoint directory")
    prune_checkpoint(parser.parse_args().checkpoint)


if __name__ == '__main__':
    main()
