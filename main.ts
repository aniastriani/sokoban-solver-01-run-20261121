// Room-by-room push-optimal Sokoban solver.
//
// The levels this answers are built from a one-cell hallway with small rooms
// (at most 6 by 6, three or four boxes each) opening off it. A box pushed out
// of a room into the hallway can never come back, so every room is solved on
// its own and the answers are stitched together: the player walks along the
// hallway from one room to the next, and each room is finished with the fewest
// pushes possible. Inside a room the search is A* over pushes, with the
// player's reachable floor as part of the state, an assignment heuristic over
// shortest-path distances, and two sound prunes (dead squares, frozen 2x2
// blocks). Everything is budgeted by a fixed number of expanded states, never
// by the clock, so two runs write the same bytes.
//
// Runs as node main.ts on Node 22 (types are stripped natively). Reads
// $MAKEFUN_INPUT_DIR/<id>/level.txt, writes
// $MAKEFUN_OUTPUT_DIR/<id>/solution.txt with one LURD letter per step
// (upper case marks a push; the checker reads the case either way).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// One step in each direction: the row/col delta, the walk letter and the push letter.
const STEPS = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
] as const;

// How far the searches may go, counted in expanded states. Fixed counts keep
// the program deterministic: no wall-clock cutoffs anywhere.
const ROOM_ASTAR_CAP = 600_000;
const ROOM_GREEDY_CAP = 600_000;
const WHOLE_MOVE_ASTAR_CAP = 200_000;

interface Board {
  w: number;
  h: number;
  wall: Uint8Array; // 1 where the player and boxes can never stand
  goals: number[]; // goal cells, sorted
  boxes: number[]; // box cells, sorted
  player: number; // the player's cell
}

interface Room {
  cells: Uint8Array; // 1 on every floor cell of this room
  boxes: number[];
  goals: number[];
}

/** Read a level in XSB. Short rows and the outer frame count as walls. */
function readBoard(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The cells a box can stand on and still be pushed onto some goal, found by
 * pulling back from the goals: a cell is live when a box there can be pushed
 * one step towards a live cell with the player standing behind it. Hallway
 * cells come out dead, because pushing a box back into a room needs the
 * player on the far side, where there is only wall.
 */
function liveCells(board: Board): Uint8Array {
  const { w, wall, goals } = board;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const cell = queue[qi];
    for (const s of STEPS) {
      const d = s.dy * w + s.dx;
      const from = cell - d; // the box stands here before the push
      const behind = from - d; // the player stands here to make the push
      if (wall[from] === 0 && wall[behind] === 0 && live[from] === 0) {
        live[from] = 1;
        queue.push(from);
      }
    }
  }
  return live;
}

/**
 * Split a level into its rooms: cut the dead cells of the player's row (the
 * hallway) out of the floor, and each connected part left that holds boxes is
 * a room. Returns null when the level is not a set of rooms, so the caller
 * can fall back to solving the whole level at once.
 */
function splitRooms(board: Board): Room[] | null {
  const { w, wall, goals, boxes, player } = board;
  const live = liveCells(board);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (wall[i] === 0 && live[i] === 0) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b] === 1)) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  const stack: number[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] === 1 || cut[s] === 1 || part[s] >= 0) continue;
    const cells = new Uint8Array(wall.length);
    const id = rooms.length;
    stack.length = 0;
    stack.push(s);
    part[s] = id;
    cells[s] = 1;
    while (stack.length > 0) {
      const cell = stack.pop()!;
      for (const st of STEPS) {
        const q = cell + st.dy * w + st.dx;
        if (wall[q] === 0 && cut[q] === 0 && part[q] < 0) {
          part[q] = id;
          cells[q] = 1;
          stack.push(q);
        }
      }
    }
    rooms.push({ cells, boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]]!.boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** A small binary heap of [priority, deeper-first, insertion order, state]. */
class Heap {
  private items: [number, number, number, number][] = [];
  get size(): number {
    return this.items.length;
  }
  push(entry: [number, number, number, number]): void {
    const a = this.items;
    a.push(entry);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Heap.before(a[i]!, a[p]!)) break;
      [a[i], a[p]] = [a[p]!, a[i]!];
      i = p;
    }
  }
  pop(): [number, number, number, number] {
    const a = this.items;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && Heap.before(a[c + 1]!, a[c]!)) c++;
        if (!Heap.before(a[c]!, a[i]!)) break;
        [a[i], a[c]] = [a[c]!, a[i]!];
        i = c;
      }
    }
    return top;
  }
  // Lower priority first; on a tie the deeper state first, then the earlier
  // insertion, so the order is fully fixed and the run is deterministic.
  private static before(x: [number, number, number, number], y: [number, number, number, number]): boolean {
    return x[0] < y[0] || (x[0] === y[0] && (x[1] > y[1] || (x[1] === y[1] && x[2] < y[2])));
  }
}

/**
 * Flood the floor the player can reach from `from` with `occupied` in the
 * way. Marks the region in `seen` with `stamp` and returns its lowest cell,
 * which names the region: two states with the same boxes and the same region
 * are the same state, whatever square the player happens to stand on.
 */
function floodRegion(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const cell = stack[--n];
    if (cell < low) low = cell;
    for (const s of STEPS) {
      const q = cell + s.dy * w + s.dx;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Shortest-path distances between floor cells, ignoring boxes. */
function distanceTable(board: Board): Int32Array[] {
  const { w, wall } = board;
  const table: Int32Array[] = new Array(wall.length);
  for (let s = 0; s < wall.length; s++) {
    const dist = new Int32Array(wall.length).fill(-1);
    table[s] = dist;
    if (wall[s] === 1) continue;
    const queue: number[] = [s];
    dist[s] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const cell = queue[qi];
      for (const st of STEPS) {
        const q = cell + st.dy * w + st.dx;
        if (wall[q] === 0 && dist[q] < 0) {
          dist[q] = dist[cell] + 1;
          queue.push(q);
        }
      }
    }
  }
  return table;
}

/**
 * The cheapest way to match every box to a different goal, over shortest-path
 * distances. Each push moves one box one cell, so no solution can finish in
 * fewer pushes than this sum: the heuristic never overestimates.
 */
function assignmentEstimate(
  boxes: number[],
  goals: number[],
  dist: Int32Array[],
): number {
  const n = boxes.length;
  const order = goals.map((_, i) => i);
  let best = Number.POSITIVE_INFINITY;
  const permute = (k: number, acc: number): void => {
    if (acc >= best) return;
    if (k === n) {
      best = acc;
      return;
    }
    for (let i = k; i < n; i++) {
      [order[k], order[i]] = [order[i]!, order[k]!];
      const d = dist[boxes[k]!]![goals[order[k]!]!]!;
      if (d >= 0) permute(k + 1, acc + d);
      [order[k], order[i]] = [order[i]!, order[k]!];
    }
  };
  permute(0, 0);
  return best;
}

/** Whether the box just pushed onto `t` is stuck in a frozen 2x2 block. */
function frozenBlock(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!((wall[a] === 1 || occupied[a] === 1) && (wall[b] === 1 || occupied[b] === 1) &&
        (wall[c] === 1 || occupied[c] === 1) && (wall[d] === 1 || occupied[d] === 1))) continue;
    if ((occupied[a] === 1 && goal[a] === 0) || (occupied[b] === 1 && goal[b] === 0) ||
        (occupied[c] === 1 && goal[c] === 0) || (occupied[d] === 1 && goal[d] === 0)) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, or null. */
function walkTo(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const cell = queue[qi];
    if (cell === to) break;
    for (const s of STEPS) {
      const q = cell + s.dy * w + s.dx;
      if (wall[q] === 1 || occupied[q] === 1 || prev[q] !== -1) continue;
      prev[q] = cell;
      how[q] = s.walk;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let out = '';
  for (let cell = to; cell !== from; cell = prev[cell]!) out = how[cell]! + out;
  return out;
}

/** Where the player ends up after playing `moves` from `from`. */
function endSquare(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    p += ch === 'u' || ch === 'U' ? -w : ch === 'd' || ch === 'D' ? w : ch === 'l' || ch === 'L' ? -1 : 1;
  }
  return p;
}

interface PushSolution {
  pushes: number;
  moves: string;
}

/**
 * A push-optimal solution for one board: A* over pushes, where a state is
 * the boxes plus the player's reachable region. `needExit` asks that the
 * player can still walk out of `roomCells` once the boxes sit on their goals,
 * so the next room stays reachable. Returns null past `cap` expansions.
 */
function solvePushes(
  board: Board,
  roomCells: Uint8Array | null,
  needExit: boolean,
  cap: number,
): PushSolution | null {
  const { w, wall, goals } = board;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveCells(board);
  if (board.boxes.some((b) => live[b] === 0)) return null;
  const dist = distanceTable(board);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const nextSeen = new Int32Array(size);
  const exitSeen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b] === 1);
  // Whether the player, standing on `player` with `occupied` marking the
  // boxes, can walk to a floor cell outside the room. The caller marks the
  // boxes it means before calling. Floods its own array: the current
  // state's region marks in `seen` must survive the call.
  const canLeave = (player: number): boolean => {
    if (roomCells === null) return true;
    floodRegion(w, wall, occupied, player, exitSeen, ++stamp, stack);
    const mark = stamp;
    for (let i = 0; i < size; i++) {
      if (exitSeen[i] === mark && roomCells[i] === 0) return true;
    }
    return false;
  };

  const stateBoxes: number[][] = [board.boxes.slice()];
  const statePlayer: number[] = [board.player];
  const stateCost: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const best = new Map<string, number>();
  for (const b of board.boxes) occupied[b] = 1;
  const startLow = floodRegion(w, wall, occupied, board.player, seen, ++stamp, stack);
  for (const b of board.boxes) occupied[b] = 0;
  best.set(`${startLow}|${board.boxes.join(',')}`, 0);
  const heap = new Heap();
  let order = 0;
  heap.push([assignmentEstimate(board.boxes, goals, dist), 0, order++, 0]);
  let expanded = 0;
  let found = -1;
  if (solved(board.boxes)) {
    if (!needExit) {
      found = 0;
    } else {
      for (const b of board.boxes) occupied[b] = 1;
      if (canLeave(board.player)) found = 0;
      for (const b of board.boxes) occupied[b] = 0;
    }
  }

  while (heap.size > 0 && found < 0) {
    const [, g, , s] = heap.pop();
    const boxes = stateBoxes[s]!;
    if (stateCost[s] !== g) continue;
    if (expanded >= cap) return null;
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    const regionStamp = ++stamp;
    floodRegion(w, wall, occupied, statePlayer[s]!, seen, regionStamp, stack);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const box = boxes[bi]!;
      for (const st of STEPS) {
        const d = st.dy * w + st.dx;
        const target = box + d;
        // The player must stand behind the box, and the box must land on a
        // live floor cell that no box already holds.
        if (seen[box - d] !== regionStamp || wall[target] === 1 || occupied[target] === 1 || live[target] === 0) continue;
        occupied[box] = 0;
        occupied[target] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, target);
        let next: number[] | null = null;
        let nextKey = '';
        let h = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = target;
          next.sort((a, b) => a - b);
          // The next state's region is flooded into its own array: stamping
          // it into `seen` would wipe the region this state is expanding.
          const low = floodRegion(w, wall, occupied, box, nextSeen, ++stamp, stack);
          nextKey = `${low}|${next.join(',')}`;
          h = assignmentEstimate(next, goals, dist);
        }
        occupied[target] = 0;
        occupied[box] = 1;
        if (frozen || next === null || h === Number.POSITIVE_INFINITY) continue;
        const known = best.get(nextKey);
        if (known !== undefined && known <= g + 1) continue;
        best.set(nextKey, g + 1);
        const t = stateBoxes.length;
        stateBoxes.push(next);
        statePlayer.push(box);
        stateCost.push(g + 1);
        parent.push(s);
        pushFrom.push(box);
        pushTo.push(target);
        if (solved(next)) {
          // The goal state's boxes differ from this state's by the one push:
          // mark the goal state's boxes before asking whether the player
          // can still walk out, then put this state's boxes back.
          let leaves = true;
          if (needExit) {
            for (const b of boxes) occupied[b] = 0;
            for (const b of next) occupied[b] = 1;
            leaves = canLeave(box);
            for (const b of next) occupied[b] = 0;
            for (const b of boxes) occupied[b] = 1;
          }
          if (leaves) {
            found = t;
            break;
          }
        }
        heap.push([g + 1 + h, g + 1, order++, t]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Walk the parent chain back and spell out the moves: the walk to each
  // push's square, then the push itself.
  const chain: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = board.player;
  let boxes = board.boxes.slice();
  for (const s of chain) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const st = STEPS.find((x) => x.dy * w + x.dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + st.push;
    player = from;
    boxes = stateBoxes[s]!.slice();
  }
  return { pushes: chain.length, moves };
}

/**
 * A greedy push search: same states as solvePushes, but ordered by the
 * heuristic alone, so it usually finds a solution where the optimal search
 * runs out of budget. Not optimal, only a safety net.
 */
function solvePushesGreedy(
  board: Board,
  roomCells: Uint8Array | null,
  needExit: boolean,
  cap: number,
): PushSolution | null {
  const { w, wall, goals } = board;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveCells(board);
  if (board.boxes.some((b) => live[b] === 0)) return null;
  const dist = distanceTable(board);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const nextSeen = new Int32Array(size);
  const exitSeen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b] === 1);
  const canLeave = (player: number): boolean => {
    if (roomCells === null) return true;
    // Its own array: the current state's region marks in `seen` must survive.
    floodRegion(w, wall, occupied, player, exitSeen, ++stamp, stack);
    const mark = stamp;
    for (let i = 0; i < size; i++) {
      if (exitSeen[i] === mark && roomCells[i] === 0) return true;
    }
    return false;
  };

  const stateBoxes: number[][] = [board.boxes.slice()];
  const statePlayer: number[] = [board.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const seenKeys = new Set<string>();
  for (const b of board.boxes) occupied[b] = 1;
  const startLow = floodRegion(w, wall, occupied, board.player, seen, ++stamp, stack);
  for (const b of board.boxes) occupied[b] = 0;
  seenKeys.add(`${startLow}|${board.boxes.join(',')}`);
  const heap = new Heap();
  let order = 0;
  heap.push([assignmentEstimate(board.boxes, goals, dist), 0, order++, 0]);
  let expanded = 0;
  let found = -1;
  if (solved(board.boxes)) {
    if (!needExit) found = 0;
    else {
      for (const b of board.boxes) occupied[b] = 1;
      if (canLeave(board.player)) found = 0;
      for (const b of board.boxes) occupied[b] = 0;
    }
  }

  while (heap.size > 0 && found < 0) {
    const [, , , s] = heap.pop();
    const boxes = stateBoxes[s]!;
    if (expanded >= cap) return null;
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    const regionStamp = ++stamp;
    floodRegion(w, wall, occupied, statePlayer[s]!, seen, regionStamp, stack);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const box = boxes[bi]!;
      for (const st of STEPS) {
        const d = st.dy * w + st.dx;
        const target = box + d;
        if (seen[box - d] !== regionStamp || wall[target] === 1 || occupied[target] === 1 || live[target] === 0) continue;
        occupied[box] = 0;
        occupied[target] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, target);
        let next: number[] | null = null;
        let nextKey = '';
        let h = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = target;
          next.sort((a, b) => a - b);
          // The next state's region is flooded into its own array: stamping
          // it into `seen` would wipe the region this state is expanding.
          const low = floodRegion(w, wall, occupied, box, nextSeen, ++stamp, stack);
          nextKey = `${low}|${next.join(',')}`;
          h = assignmentEstimate(next, goals, dist);
        }
        occupied[target] = 0;
        occupied[box] = 1;
        if (frozen || next === null || h === Number.POSITIVE_INFINITY || seenKeys.has(nextKey)) continue;
        seenKeys.add(nextKey);
        const t = stateBoxes.length;
        stateBoxes.push(next);
        statePlayer.push(box);
        parent.push(s);
        pushFrom.push(box);
        pushTo.push(target);
        if (solved(next)) {
          let leaves = true;
          if (needExit) {
            for (const b of boxes) occupied[b] = 0;
            for (const b of next) occupied[b] = 1;
            leaves = canLeave(box);
            for (const b of next) occupied[b] = 0;
            for (const b of boxes) occupied[b] = 1;
          }
          if (leaves) {
            found = t;
            break;
          }
        }
        heap.push([h, 0, order++, t]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  const chain: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = board.player;
  let boxes = board.boxes.slice();
  for (const s of chain) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const st = STEPS.find((x) => x.dy * w + x.dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + st.push;
    player = from;
    boxes = stateBoxes[s]!.slice();
  }
  return { pushes: chain.length, moves };
}

/**
 * A last-resort solver over single steps (not pushes): A* where a state is
 * the player and the boxes, each step costs one, and the estimate is each
 * box's Manhattan distance to its nearest goal, summed. It finds fewest-move
 * answers, which are rarely fewest-push, but a solved level still scores.
 */
function solveByMoves(board: Board, cap: number): PushSolution | null {
  const { w, wall, goals } = board;
  const goal = new Uint8Array(wall.length);
  for (const g of goals) goal[g] = 1;
  const manhattan = (b: number): number =>
    Math.min(...goals.map((g) => Math.abs((g % w) - (b % w)) + Math.abs(Math.floor(g / w) - Math.floor(b / w))));
  const estimate = (boxes: number[]): number => boxes.reduce((a, b) => a + manhattan(b), 0);
  const keyOf = (p: number, boxes: number[]): string => `${p}|${boxes.join(',')}`;
  const players: number[] = [board.player];
  const boxesList: number[][] = [board.boxes.slice()];
  const costs: number[] = [0];
  const parent: number[] = [-1];
  const stepOf: string[] = [''];
  const best = new Map<string, number>([[keyOf(board.player, board.boxes), 0]]);
  const heap = new Heap();
  let order = 0;
  heap.push([estimate(board.boxes), 0, order++, 0]);
  let expanded = 0;
  while (heap.size > 0) {
    const [, g, , s] = heap.pop();
    if (costs[s] !== g) continue;
    const boxes = boxesList[s]!;
    if (boxes.every((b) => goal[b] === 1)) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = stepOf[t]! + moves;
      return { pushes: -1, moves };
    }
    if (expanded >= cap) return null;
    expanded++;
    const p = players[s]!;
    for (const st of STEPS) {
      const d = st.dy * w + st.dx;
      const q = p + d;
      if (wall[q] === 1) continue;
      let next = boxes;
      let letter = st.walk;
      const at = boxes.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] === 1 || boxes.includes(q + d)) continue;
        next = boxes.slice();
        next[at] = q + d;
        next.sort((a, b) => a - b);
        letter = st.push;
      }
      const key = keyOf(q, next);
      const known = best.get(key);
      if (known !== undefined && known <= g + 1) continue;
      best.set(key, g + 1);
      const t = players.length;
      players.push(q);
      boxesList.push(next);
      costs.push(g + 1);
      parent.push(s);
      stepOf.push(letter);
      heap.push([g + 1 + estimate(next), g + 1, order++, t]);
    }
  }
  return null;
}

/**
 * Solve one level: split it into rooms and solve each room push-optimally,
 * in room order, with the other rooms' boxes standing as walls; the player
 * walks on from where the last room left it. Falls back, per room, to an
 * optimal search without the exit condition, then to a greedy push search,
 * and for the whole level to the step-by-step solver. An empty answer when
 * nothing solves it.
 */
function solveLevel(text: string): string {
  const level = readBoard(text);
  const rooms = splitRooms(level);
  if (rooms === null) {
    const whole = solvePushes(level, null, false, ROOM_ASTAR_CAP)
      ?? solveByMoves(level, WHOLE_MOVE_ASTAR_CAP);
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const last = ri === rooms.length - 1;
    // The other rooms' boxes are walls: solved ones where their goals are,
    // unsolved ones where they started.
    const wall = level.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const other = rooms[oj]!;
      const at = oj < ri ? other.goals : other.boxes;
      for (const b of at) wall[b] = 1;
    }
    const sub: Board = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushes(sub, room.cells, !last, ROOM_ASTAR_CAP)
      ?? solvePushes(sub, room.cells, false, ROOM_ASTAR_CAP)
      ?? solvePushesGreedy(sub, room.cells, !last, ROOM_GREEDY_CAP)
      ?? solvePushesGreedy(sub, room.cells, false, ROOM_GREEDY_CAP);
    if (solved === null) {
      const whole = solveByMoves(level, WHOLE_MOVE_ASTAR_CAP);
      return whole === null ? '' : whole.moves;
    }
    moves += solved.moves;
    player = endSquare(level.w, player, solved.moves);
  }
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
