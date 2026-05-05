---
name: Groupby rolling smoothing scope
description: Don't suggest cycle_id in groupby for rolling smoothing when data is already filtered to be contiguous
type: feedback
---

Don't change `groupby(["id"])` to `groupby(["id", "cycle_id"])` for rolling averages on this project.

**Why:** Cycles are contiguous after the filtering step in dataset.py, so bleeding across cycle boundaries within a subject is not a concern.

**How to apply:** Leave `["id"]`-only groupby for rolling/smoothing transforms unless the user explicitly raises cross-cycle bleeding as a problem.
