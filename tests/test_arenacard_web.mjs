import assert from "node:assert/strict";
import test from "node:test";
import { arenaText } from "../web/js/arenacard.js";

const row = (model, elo, w, l, t) => ({model, elo, wins: w, losses: l, ties: t, games: w + l + t});
const d = {gpus: ["RTX 5080", "RTX 5060 Ti"], games: 31,
  board: [row("qwen3.6-35b-a3b-ud-q6-k", 1071, 12, 4, 2), row("gemma-4-12b", 1002, 9, 9, 1),
          row("a", 990, 1, 2, 0), row("b", 980, 1, 2, 0), row("c", 970, 1, 2, 0), row("d", 960, 0, 3, 0),
          row("e", 950, 0, 3, 0)]};

test("every string on the card", () => {
  const t = arenaText(d);
  assert.equal(t.kicker, "MY BLIND ARENA");
  assert.equal(t.sub, "RTX 5080 + RTX 5060 Ti  ·  31 blind votes  ·  my prompts");
  assert.deepEqual(t.rows[0], {rank: "1", name: "qwen3.6-35b-a3b-ud-q6-k", elo: "1071", wlt: "12-4-2"});
  assert.equal(t.rows.length, 6);                 // the top six fit
  assert.equal(t.more, "+1 more");
  assert.equal(t.foot, "github.com/dadwritestech/LlamaForge");
});

test("one vote reads as one, and an empty board still draws", () => {
  assert.match(arenaText({gpus: [], games: 1, board: [row("x", 1016, 1, 0, 0)]}).sub, /^1 blind vote  ·/);
  const t = arenaText({});
  assert.deepEqual(t.rows, []);
  assert.equal(t.more, "");
  assert.match(t.empty, /No votes yet/);
});
