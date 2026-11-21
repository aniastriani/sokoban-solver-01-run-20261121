# Sokoban solver

A push-optimal Sokoban solver for make.fun's "Build the best Sokoban
solver, 21 November 2026" bounty.

## How it works

Each level is a one-cell hallway with small rooms (at most 6 by 6, three or
four boxes each) opening off it. A box pushed out of a room into the hallway
can never come back, so the solver splits a level into its rooms, solves
each room on its own with the fewest pushes possible, and stitches the
answers together while the player walks along the hallway from room to room.

Inside a room the search is A* over pushes. A state is the boxes plus the
player's reachable floor; the heuristic is the cheapest assignment of boxes
to goals over shortest-path distances, which never overestimates, so the
first solution found uses the fewest pushes. Two sound prunes keep the
search small: dead squares (a box that can never reach a goal) and frozen
2x2 blocks. If the optimal search runs out of its fixed state budget, the
solver falls back to a greedy push search and then to a step-by-step A*,
so a level still gets answered. All budgets are fixed counts of expanded
states, never the clock, so two runs write the same bytes.

## Run it

```sh
node main.ts
```

It reads `$MAKEFUN_INPUT_DIR/<id>/level.txt` (default `/input`) and writes
`$MAKEFUN_OUTPUT_DIR/<id>/solution.txt` (default `/output`) with one LURD
letter per step; an upper-case letter marks a push.

## Practice score

1,000,000 out of 1,000,000 on the kit's practice set: every level solved in
its push-optimal count. Three further sets drawn from the kit's generator
with fresh seeds also score 1,000,000 each.

Built by an AI agent for make.fun. See https://make.fun/bounty/sokoban-solver-01-run-20261121.
