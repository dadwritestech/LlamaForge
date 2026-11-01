// web/js/qr.js: the Phone dialog's QR encoder. Round-tripped through a real
// decoder (jsQR) when it was written; these pin the parts that can't drift.
import { test } from "node:test";
import assert from "node:assert/strict";
import { qrMatrix, qrSVG, rsEncode } from "../web/js/qr.js";

test("Reed-Solomon matches the published HELLO WORLD 1-M vector", () => {
  const data = [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17];
  assert.deepEqual(rsEncode(data, 10), [196, 35, 39, 119, 235, 215, 231, 226, 93, 23]);
});

test("picks the smallest version that fits, up to v10", () => {
  assert.equal(qrMatrix("x").length, 21);                       // v1
  assert.equal(qrMatrix("http://192.168.1.20:8092/#k=" + "a".repeat(22)).length, 33);  // v4: 50 bytes
  assert.equal(qrMatrix("http://192.168.1.20:8092/#k=" + "a".repeat(43)).length, 37);  // v5: 71 > 64
  assert.equal(qrMatrix("q".repeat(213)).length, 57);           // v10
  assert.equal(qrMatrix("q".repeat(214)), null);
});

test("finder patterns, timing and the dark module are where scanners look", () => {
  const m = qrMatrix("http://192.168.1.20:8092/");
  const n = m.length;
  for (const [cx, cy] of [[3, 3], [n - 4, 3], [3, n - 4]]) {
    assert.equal(m[cy][cx], true);
    assert.equal(m[cy - 2][cx], false);
    assert.equal(m[cy - 3][cx], true);
  }
  for (let i = 8; i < n - 8; i++) assert.equal(m[6][i], i % 2 === 0);
  assert.equal(m[n - 8][8], true);
});

test("the SVG is built from the matrix only", () => {
  const svg = qrSVG("<script>alert(1)</script>");
  assert.match(svg, /^<svg /);
  assert.doesNotMatch(svg, /script/);
});
