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
"""Pick the rollout reassembler class, the way call_selection picks a policy.

``token_capture.reassembler`` is a built-in name or ``package.module:Class``.
The config only checks the shape of the name, because importing a reassembler
pulls in the data plane. Setup and the reassembler actors resolve it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler

REASSEMBLER_CLASSES: dict[str, str] = {
    "default": "nemo_rl.experience.rollout_reassembler:RolloutReassembler",
    "shared_prefix": (
        "nemo_rl.experience.shared_prefix_reassembler:SharedPrefixRolloutReassembler"
    ),
}


def check_reassembler_name(name: str) -> None:
    """Reject a name that is neither a built-in nor ``package.module:Class``."""
    if name in REASSEMBLER_CLASSES:
        return
    module_name, separator, class_name = name.partition(":")
    if not separator or not module_name or not class_name.isidentifier():
        raise ValueError(
            f"Unknown token_capture.reassembler {name!r}: expected a built-in "
            f"({', '.join(REASSEMBLER_CLASSES)}) or 'package.module:Class'"
        )


def resolve_reassembler_class(name: str) -> type[RolloutReassembler]:
    """Import the named reassembler class and check that it is one."""
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler

    check_reassembler_name(name)
    module_name, _, class_name = REASSEMBLER_CLASSES.get(name, name).partition(":")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(
            f"Cannot load token_capture.reassembler {name!r}: {error}"
        ) from error
    if not (isinstance(cls, type) and issubclass(cls, RolloutReassembler)):
        raise ValueError(
            f"token_capture.reassembler {name!r} is not a RolloutReassembler subclass"
        )
    return cls
