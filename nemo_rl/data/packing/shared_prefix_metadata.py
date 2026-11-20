# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared-prefix row metadata carried through NeMo RL batches.

The rollout side describes which rows share which tokens as a small tree per
prompt group (:func:`build_prefix_tree`). That description has no backend in
it: any trainer can read it, and a trainer that ignores it runs dense rows,
which is always correct. Physical packing is owned by :mod:`megatron.rl`;
callers import it lazily so driver-side modules that only route batches stay
importable without the optional Megatron installation.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

__all__ = [
    "SHARED_PREFIX_GROUP_ID",
    "SHARED_PREFIX_PARENT",
    "SHARED_PREFIX_PROMPT_LENGTHS",
    "SHARED_PREFIX_SHARED_LEN",
    "SHARED_PREFIX_SLOT",
    "SHARED_PREFIX_TREE_COLUMNS",
    "build_prefix_tree",
    "prefix_tree_share_keys",
    "with_prompt_length_tags",
]

SHARED_PREFIX_GROUP_ID = "shared_prefix_group_id"
SHARED_PREFIX_PROMPT_LENGTHS = "shared_prefix_prompt_lengths"
# The row's position in its prompt group, as the reassembler published it.
SHARED_PREFIX_SLOT = "shared_prefix_slot"
# The slot of the earlier row this row shares the most leading tokens with,
# or -1 when it shares none.
SHARED_PREFIX_PARENT = "shared_prefix_parent"
# How many leading token IDs this row has in common with its parent row.
SHARED_PREFIX_SHARED_LEN = "shared_prefix_shared_len"
# Every column a prefix-tree reassembler publishes. The prompt length marks
# where each row's own context ends.
SHARED_PREFIX_TREE_COLUMNS = (
    SHARED_PREFIX_PROMPT_LENGTHS,
    SHARED_PREFIX_SLOT,
    SHARED_PREFIX_PARENT,
    SHARED_PREFIX_SHARED_LEN,
)


def _common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    """Count the leading tokens two rows share, stopping near the first difference.

    Rows in a group usually differ right after the prompt, so comparing in
    doubling chunks costs about the shared length, not the full row length.
    """
    limit = min(len(a), len(b))
    start, step = 0, 256
    while start < limit:
        end = min(limit, start + step)
        differ = np.flatnonzero(a[start:end] != b[start:end])
        if differ.size:
            return start + int(differ[0])
        start, step = end, step * 2
    return limit


def build_prefix_tree(
    token_ids: Sequence[Sequence[int]],
    valid: Sequence[bool],
) -> tuple[list[int], list[int]]:
    """Link each valid row to the earlier valid row it shares the most tokens with.

    Rows are one prompt group, in publication order. Invalid rows (capture
    placeholders, forest execution padding) never get a parent and are never
    one. Ties go to the earliest row, so the result does not depend on how
    many rows tie.

    Args:
        token_ids: Each row's real token IDs, without padding.
        valid: Whether each row is a real, verified row.

    Returns:
        ``(parents, shared_lens)``: per row, the parent slot (-1 for none) and
        the number of leading tokens it shares with that parent (0 for none).
    """
    if len(token_ids) != len(valid):
        raise ValueError("token_ids and valid must have one entry per row")
    rows = [np.asarray(ids, dtype=np.int64) for ids in token_ids]
    parents: list[int] = []
    shared_lens: list[int] = []
    for index, row in enumerate(rows):
        best_parent, best_len = -1, 0
        if valid[index]:
            for earlier in range(index):
                if not valid[earlier]:
                    continue
                common = _common_prefix_len(row, rows[earlier])
                if common > best_len:
                    best_parent, best_len = earlier, common
        parents.append(best_parent)
        shared_lens.append(best_len)
    return parents, shared_lens


def prefix_tree_share_keys(
    group_ids: Sequence[Hashable],
    slots: Sequence[int],
    parents: Sequence[int],
) -> list[str]:
    """Key each row by its group and the root of its prefix tree.

    Rows with the same key are the ones the rollout side says may share a
    prefix. A row whose parent is not in this batch is treated as a root.
    Keys are strings, ``"<group>#<root slot>"``, so they can stand in for
    group IDs anywhere a group ID is expected.
    """
    if not len(group_ids) == len(slots) == len(parents):
        raise ValueError("group_ids, slots and parents must be row-aligned")
    parent_of: dict[tuple[Hashable, int], int] = {}
    for group, slot, parent in zip(group_ids, slots, parents, strict=True):
        key = (group, int(slot))
        if key in parent_of:
            raise ValueError(f"duplicate prefix-tree slot {key!r}")
        if parent >= slot:
            raise ValueError("a prefix-tree parent must be an earlier row")
        parent_of[key] = int(parent)
    keys = []
    for group, slot in zip(group_ids, slots, strict=True):
        node = int(slot)
        while parent_of.get((group, node), -1) >= 0 and (
            (group, parent_of[(group, node)]) in parent_of
        ):
            node = parent_of[(group, node)]
        keys.append(f"{group}#{node}")
    return keys


def with_prompt_length_tags(
    tags: Sequence[Mapping[str, Any]] | None,
    *,
    prompt_lengths: Sequence[int],
    sequence_lengths: Sequence[int],
) -> list[dict[str, Any]]:
    """Carry prompt lengths through the metadata's existing row transforms."""
    if len(prompt_lengths) != len(sequence_lengths):
        raise ValueError("Prompt lengths must align with sequence lengths")
    if tags is not None and len(tags) != len(prompt_lengths):
        raise ValueError("Prompt lengths must align with row tags")
    result = (
        [dict(tag) for tag in tags]
        if tags is not None
        else [{} for _ in prompt_lengths]
    )
    for tag, prefix, length in zip(
        result, prompt_lengths, sequence_lengths, strict=True
    ):
        if type(prefix) is not int or not 0 <= prefix <= length:
            raise ValueError(
                "Prompt length must be an integer within the real sequence"
            )
        if (
            SHARED_PREFIX_PROMPT_LENGTHS in tag
            and tag[SHARED_PREFIX_PROMPT_LENGTHS] != prefix
        ):
            raise ValueError("Prompt length tag disagrees with its tensor field")
        tag[SHARED_PREFIX_PROMPT_LENGTHS] = prefix
    return result


def shared_prefix_columns(
    token_ids: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    valid: Sequence[bool],
) -> dict[str, torch.Tensor]:
    """Build one prompt group's prefix-tree columns, one value per row.

    The rollout reassembler writes these on every row, whatever the trainer's
    packing. Only shared-prefix packing reads them today; other consumers
    (value model, teachers, dense packing) ignore them. They cost four int64s
    per row, so there is no switch.

    Placeholders and forest execution padding (``valid`` False) get prompt
    length 0 and no parent, so they always run dense.
    """
    parents, shared_lens = build_prefix_tree(token_ids, valid)
    return {
        SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor(
            [int(n) if ok else 0 for n, ok in zip(prompt_lens, valid, strict=True)],
            dtype=torch.long,
        ),
        SHARED_PREFIX_SLOT: torch.arange(len(token_ids), dtype=torch.long),
        SHARED_PREFIX_PARENT: torch.tensor(parents, dtype=torch.long),
        SHARED_PREFIX_SHARED_LEN: torch.tensor(shared_lens, dtype=torch.long),
    }
