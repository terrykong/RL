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
"""The backend-neutral prefix tree a shared-prefix reassembler publishes."""

from dataclasses import dataclass

import pytest
import torch

from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_PARENT,
    SHARED_PREFIX_PROMPT_LENGTHS,
    SHARED_PREFIX_SHARED_LEN,
    SHARED_PREFIX_SLOT,
    SHARED_PREFIX_TREE_COLUMNS,
    build_prefix_tree,
    prefix_tree_share_keys,
)

P = [7, 7, 7, 7]  # the group's prompt
A = [1, 1]  # a model call that two branches continue


def test_ordinary_group_links_every_sibling_to_the_first_row():
    rows = [P + [5, 6], P + [8], P + [9, 9, 9]]
    parents, shared = build_prefix_tree(rows, [True, True, True])
    assert parents == [-1, 0, 0]
    assert shared == [0, len(P), len(P)]
    assert prefix_tree_share_keys(["g"] * 3, [0, 1, 2], parents) == ["g#0"] * 3


def test_forest_group_branches_share_the_parent_call_and_compaction_is_a_new_root():
    # Rollout 1 branches after call A; rollout 2 compacts, so its second row
    # starts from a summary and shares nothing.
    rows = [
        P + A + [2, 2],  # r0: path A -> B
        P + A + [3],  # r1: path A -> C
        P + [4, 4, 4],  # r2: rollout 2 before compaction
        [6, 6] + [5],  # r3: rollout 2 after compaction
    ]
    parents, shared = build_prefix_tree(rows, [True] * 4)
    assert parents == [-1, 0, 0, -1]
    assert shared == [0, len(P + A), len(P), 0]
    assert prefix_tree_share_keys(["g"] * 4, [0, 1, 2, 3], parents) == [
        "g#0",
        "g#0",
        "g#0",
        "g#3",
    ]


def test_invalid_rows_never_link_and_are_never_linked_to():
    rows = [P + [5], P + [5], P + [6]]
    parents, shared = build_prefix_tree(rows, [False, True, True])
    assert parents == [-1, -1, 1]
    assert shared == [0, 0, len(P)]


def test_share_keys_follow_chains_to_the_root_and_treat_missing_parents_as_roots():
    keys = prefix_tree_share_keys(["g", "g", "g", "h"], [0, 1, 2, 3], [-1, 0, 1, 1])
    assert keys == ["g#0", "g#0", "g#0", "h#3"]


def test_share_keys_reject_a_parent_that_is_not_earlier():
    with pytest.raises(ValueError, match="earlier row"):
        prefix_tree_share_keys(["g", "g"], [0, 1], [1, -1])


@dataclass
class _Row:
    token_ids: list[int]
    prompt_len: int
    valid: bool


def test_shared_prefix_reassembler_publishes_exactly_the_tree_columns():
    from nemo_rl.experience.reassembler_registry import resolve_reassembler_class

    cls = resolve_reassembler_class("shared_prefix")
    assert cls.extra_row_columns == SHARED_PREFIX_TREE_COLUMNS
    rows = [_Row(P + [5], 4, True), _Row(P + [6], 4, True), _Row(P, 0, False)]
    # _row_extras reads only its argument, so no data-plane client is needed.
    extras = cls._row_extras(object.__new__(cls), rows)
    assert set(extras) == set(SHARED_PREFIX_TREE_COLUMNS)
    assert extras[SHARED_PREFIX_PROMPT_LENGTHS].tolist() == [4, 4, 0]
    assert extras[SHARED_PREFIX_SLOT].tolist() == [0, 1, 2]
    assert extras[SHARED_PREFIX_PARENT].tolist() == [-1, 0, -1]
    assert extras[SHARED_PREFIX_SHARED_LEN].tolist() == [0, 4, 0]
    assert all(t.dtype == torch.long for t in extras.values())


def test_default_reassembler_publishes_no_extra_columns():
    from nemo_rl.experience.reassembler_registry import resolve_reassembler_class

    cls = resolve_reassembler_class("default")
    assert cls.extra_row_columns == ()
    assert cls._row_extras(object.__new__(cls), []) == {}


@pytest.mark.parametrize("name", ["nope", "pkg.mod:", ":Cls", "pkg.mod:not an id"])
def test_reassembler_names_are_checked_without_importing(name):
    from nemo_rl.experience.reassembler_registry import check_reassembler_name

    with pytest.raises(ValueError, match="Unknown token_capture.reassembler"):
        check_reassembler_name(name)


def test_a_custom_reassembler_must_subclass_rolloutreassembler():
    from nemo_rl.experience.reassembler_registry import resolve_reassembler_class

    with pytest.raises(ValueError, match="not a RolloutReassembler subclass"):
        resolve_reassembler_class("collections:OrderedDict")
