// Tetris replay viewer: loads exported games and draws them. It never simulates
// Tetris; every board comes from the Python engine (see games/tetris/frames.py
// for the frame format).
//
// A few JavaScript basics used below, for a Python programmer:
//   const x = ...      a variable that won't be reassigned (let = one that will)
//   (a, b) => a + b    a short anonymous function, like Python's lambda
//   async / await      like Python's asyncio: `await fetch(url)` waits for a
//                      download without freezing the page
//   document.getElementById("x")   finds the HTML element with id="x"
//   element.addEventListener("click", fn)   runs fn when that event happens
//   `text ${expr}`     a template string, like Python's f"text {expr}"
//   ===, !==           strict equality (no type conversion); use these, not ==
//   null               JavaScript's None

"use strict"; // catches some silent mistakes (e.g. assigning to an undeclared variable)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const CELL = 24; // pixels per square in the canvas's own resolution (CSS scales it)
const SPEEDS = [1, 2, 4, 8, 15, 30, 60, 120, 250, 500]; // pieces per second
const DEFAULT_SPEED = 8;
const FLASH_MAX_SPEED = 20; // above this, line-clear flashes would just flicker
const HIDDEN_BG = "#15161c"; // the 2 spawn rows (same as the GIFs)
const CLEAR_NAMES = { 1: "single", 2: "double", 3: "triple", 4: "TETRIS!" };

// ---------------------------------------------------------------------------
// State: everything the page currently shows lives in this one object, and
// draw() turns it into pixels. Changing what's on screen = change state, then
// call draw(). (A common pattern: easier to reason about than poking at
// the page from many places.)
// ---------------------------------------------------------------------------
const state = {
  index: [],          // entries of data/index.json (one per game)
  games: [null, null],// loaded frames documents for panel A and panel B (null = none)
  n: 0,               // current piece number, shared by both panels (that's "in sync")
  playing: false,
  speed: DEFAULT_SPEED,
  carry: 0,           // fraction of the next piece already "elapsed" during playback
  lastTime: null,     // timestamp of the previous animation frame
};
const cache = new Map(); // file name -> loaded document, so switching back is instant

// Shortcut for finding elements.
const $ = (id) => document.getElementById(id);
const panels = [$("panel-a"), $("panel-b")];

// ---------------------------------------------------------------------------
// Loading data
// ---------------------------------------------------------------------------
async function loadJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

async function loadGame(entry) {
  if (!cache.has(entry.file)) {
    const doc = await loadJSON(`data/${entry.file}`);
    // Precompute the piece numbers where lines were cleared, once, so
    // "jump to next clear" doesn't scan thousands of frames each time.
    doc.clears = doc.frames.filter((f) => f.cleared_rows.length > 0).map((f) => f.n);
    doc.entry = entry;
    cache.set(entry.file, doc);
  }
  return cache.get(entry.file);
}

function showMessage(html) {
  const box = $("message");
  box.innerHTML = html;
  box.hidden = !html;
}

// ---------------------------------------------------------------------------
// Picker (the two drop-down lists)
// ---------------------------------------------------------------------------
function entryLabel(e) {
  const star = e.best ? "  ★ best" : "";
  return `step ${e.step.toLocaleString()} · seed ${e.seed} · ` +
         `${e.pieces.toLocaleString()} pieces · ${e.lines.toLocaleString()} lines${star}`;
}

function fillPicker(select, withNone) {
  select.innerHTML = "";
  if (withNone) select.add(new Option("— none (single game) —", ""));
  // One <optgroup> (a labelled section of the list) per training run.
  const runs = [...new Set(state.index.map((e) => e.run))];
  for (const run of runs) {
    const group = document.createElement("optgroup");
    group.label = `run: ${run}`;
    for (const e of state.index.filter((x) => x.run === run)) {
      group.appendChild(new Option(entryLabel(e), e.id));
    }
    select.appendChild(group);
  }
}

async function choose(slot, id) {
  // slot 0 = panel A, 1 = panel B. An empty id clears panel B.
  const entry = state.index.find((e) => e.id === id);
  if (!entry) {
    state.games[slot] = null;
  } else {
    showMessage(`Loading ${entry.id} …`);
    state.games[slot] = await loadGame(entry);
    showMessage("");
  }
  $(slot === 0 ? "pick-a" : "pick-b").value = entry ? entry.id : "";
  panels[1].hidden = state.games[1] === null;
  setPiece(state.n); // re-clamp the position for the new game length
  saveToURL();
}

// ---------------------------------------------------------------------------
// Position helpers
// ---------------------------------------------------------------------------
const loaded = () => state.games.filter((g) => g !== null);

// Last piece number of the longest loaded game (the scrubber's maximum).
function maxPiece() {
  return Math.max(0, ...loaded().map((g) => g.frames.length - 1));
}

// The frame a game shows at piece n. A game that already ended stays on its
// last frame, which is how compare mode handles games of different lengths.
function frameAt(game, n) {
  return game.frames[Math.min(n, game.frames.length - 1)];
}

function setPiece(n) {
  state.n = Math.max(0, Math.min(n, maxPiece()));
  state.carry = 0;
  draw();
}

// Next / previous piece (after / before n) where ANY shown game clears lines.
function nextClear(n) {
  const c = loaded().flatMap((g) => g.clears).filter((x) => x > n);
  return c.length ? Math.min(...c) : null;
}
function prevClear(n) {
  const c = loaded().flatMap((g) => g.clears).filter((x) => x < n);
  return c.length ? Math.max(...c) : null;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
function draw() {
  state.games.forEach((game, slot) => {
    if (game) drawPanel(panels[slot], game, state.n);
  });
  const max = maxPiece();
  const scrub = $("scrubber");
  scrub.max = max;
  scrub.value = state.n;
  $("position").textContent = `piece ${state.n.toLocaleString()} / ${max.toLocaleString()}`;
  $("btn-play").innerHTML = state.playing ? "&#x23F8; Pause" : "&#x25B6; Play";
}

function drawPanel(panel, game, n) {
  const frame = frameAt(game, n);
  const e = game.entry;
  panel.querySelector(".panel-title").innerHTML =
    `run ${e.run}, step ${e.step.toLocaleString()}${e.best ? " ★" : ""}` +
    `<small>seed ${e.seed} · ${e.pieces.toLocaleString()} pieces · ` +
    `${e.score.toLocaleString()} points</small>`;

  // Flash: during the first half of a piece's time slot, show the board just
  // BEFORE the line clear (white full rows). Only while playing at modest
  // speed, and only on the exact piece that cleared (not on a frozen ended game).
  const isCurrent = n <= game.frames.length - 1;
  const flash = state.playing && state.speed <= FLASH_MAX_SPEED && isCurrent &&
                frame.cleared_rows.length > 0 && state.carry < 0.5;
  drawBoard(panel.querySelector("canvas"), game, frame, flash);
  drawStats(panel.querySelector(".stats"), frame, isCurrent);
}

function drawBoard(canvas, game, frame, flash) {
  const h = game.header;
  const ctx = canvas.getContext("2d"); // the "2d context" is the drawing API of a canvas
  const colors = h.piece_colors;

  // Which board to paint. Normally the frame's own board string. For a flash,
  // the previous board with the new piece painted in (pure drawing: the
  // frame tells us the cells and which rows were full).
  let board = frame.board;
  if (flash) {
    const prev = game.frames[frame.n - 1];
    const cells = prev.board.split(""); // string -> array of 220 characters
    for (const [r, c] of frame.cells) cells[r * h.width + c] = String(frame.piece);
    board = cells.join("");
  }
  const white = new Set(flash ? frame.cleared_rows : []);

  ctx.fillStyle = "#0e0f13";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < h.height; r++) {
    for (let c = 0; c < h.width; c++) {
      const v = board.charCodeAt(r * h.width + c) - 48; // '0' has character code 48
      if (white.has(r)) ctx.fillStyle = "#ffffff";
      else if (v > 0) ctx.fillStyle = colors[v];
      else ctx.fillStyle = r < h.hidden_rows ? HIDDEN_BG : colors[0];
      // -1: leave a 1px gap between squares so the grid shows
      ctx.fillRect(c * CELL, r * CELL, CELL - 1, CELL - 1);
    }
  }
  // Line under the hidden spawn rows.
  ctx.fillStyle = "#3a3d4a";
  ctx.fillRect(0, h.hidden_rows * CELL - 2, h.width * CELL, 2);

  // Outline the piece that was just placed, so you can follow the decision.
  // Skipped when rows cleared: the cells are in "before clear" coordinates
  // and everything above the cleared rows has since moved down.
  if (!flash && frame.cells.length && frame.cleared_rows.length === 0) {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    for (const [r, c] of frame.cells) ctx.strokeRect(c * CELL + 1, r * CELL + 1, CELL - 3, CELL - 3);
  }

  if (frame.game_over) {
    const y = (h.height * CELL) / 2;
    ctx.fillStyle = "rgba(179, 38, 30, 0.92)";
    ctx.fillRect(0, y - 24, canvas.width, 48);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 24px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("GAME OVER", canvas.width / 2, y);
  }
}

function drawStats(table, frame, isCurrent) {
  const rows = [
    ["Piece", frame.n.toLocaleString()],
    ["Score", frame.score.toLocaleString()],
    ["Lines", frame.lines.toLocaleString()],
    ["Stack height", frame.height],
    ["Holes", frame.holes],
    ["Bumpiness", frame.bumpiness],
  ];
  let html = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  const k = frame.cleared_rows.length;
  // Keep the row's space even when empty, so the panel doesn't jump in height.
  const event = !isCurrent ? "(game over)" : k ? `cleared a ${CLEAR_NAMES[k]}` : " ";
  html += `<tr class="event"><td colspan="2">${event}</td></tr>`;
  table.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Playback: requestAnimationFrame calls `tick` before every screen repaint
// (~60 times a second). We advance by (elapsed seconds x speed) pieces. At
// high speeds that's several pieces per repaint; we only draw the last one.
// ---------------------------------------------------------------------------
function tick(time) {
  if (!state.playing) return;
  if (state.lastTime !== null) {
    state.carry += ((time - state.lastTime) / 1000) * state.speed;
    const whole = Math.floor(state.carry);
    if (whole > 0) {
      state.n = Math.min(state.n + whole, maxPiece());
      state.carry -= whole;
    }
    if (state.n >= maxPiece()) {
      state.playing = false; // reached the end
      state.carry = 0;
      saveToURL();
    }
    draw();
  }
  state.lastTime = time;
  if (state.playing) requestAnimationFrame(tick);
}

function setPlaying(on) {
  if (on && state.n >= maxPiece()) state.n = 0; // pressing play at the end restarts
  state.playing = on;
  state.lastTime = null;
  state.carry = 0;
  if (on) requestAnimationFrame(tick);
  else saveToURL();
  draw();
}

function changeSpeed(direction) {
  const i = SPEEDS.indexOf(state.speed) + direction;
  if (i >= 0 && i < SPEEDS.length) {
    state.speed = SPEEDS[i];
    $("speed").value = String(state.speed);
  }
}

// ---------------------------------------------------------------------------
// The URL "hash" (the part after #) remembers what you're looking at, e.g.
//   index.html#a=lr_decay/step000086944_seed10100&b=...&n=250
// so you can bookmark or share a moment of a game.
// ---------------------------------------------------------------------------
function saveToURL() {
  const p = new URLSearchParams();
  if (state.games[0]) p.set("a", state.games[0].entry.id);
  if (state.games[1]) p.set("b", state.games[1].entry.id);
  if (state.n) p.set("n", state.n);
  // replaceState changes the address bar without reloading or adding a
  // "back" history entry for every piece.
  history.replaceState(null, "", "#" + p.toString());
}

function readURL() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { a: p.get("a"), b: p.get("b"), n: Number(p.get("n") || 0) };
}

// ---------------------------------------------------------------------------
// Wiring: connect buttons, lists and keys to the functions above.
// ---------------------------------------------------------------------------
function wireControls() {
  $("btn-play").addEventListener("click", () => setPlaying(!state.playing));
  $("btn-back").addEventListener("click", () => setPiece(state.n - 1));
  $("btn-fwd").addEventListener("click", () => setPiece(state.n + 1));
  $("btn-start").addEventListener("click", () => setPiece(0));
  $("btn-end").addEventListener("click", () => setPiece(maxPiece()));
  $("btn-next-clear").addEventListener("click", () => {
    const c = nextClear(state.n);
    if (c !== null) setPiece(c);
  });
  $("btn-prev-clear").addEventListener("click", () => {
    const c = prevClear(state.n);
    if (c !== null) setPiece(c);
  });
  // "input" fires continuously while the slider is dragged.
  $("scrubber").addEventListener("input", (ev) => setPiece(Number(ev.target.value)));
  $("scrubber").addEventListener("change", saveToURL);

  const speed = $("speed");
  for (const s of SPEEDS) speed.add(new Option(`${s} / s`, String(s)));
  speed.value = String(state.speed);
  speed.addEventListener("change", () => { state.speed = Number(speed.value); });

  $("pick-a").addEventListener("change", (ev) => choose(0, ev.target.value));
  $("pick-b").addEventListener("change", (ev) => choose(1, ev.target.value));

  document.addEventListener("keydown", (ev) => {
    // Let drop-down lists keep their own arrow keys.
    if (ev.target.tagName === "SELECT") return;
    const step = ev.shiftKey ? 10 : 1;
    const actions = {
      " ": () => setPlaying(!state.playing),
      ArrowRight: () => setPiece(state.n + step),
      ArrowLeft: () => setPiece(state.n - step),
      Home: () => setPiece(0),
      End: () => setPiece(maxPiece()),
      n: () => $("btn-next-clear").click(),
      N: () => $("btn-next-clear").click(),
      p: () => $("btn-prev-clear").click(),
      P: () => $("btn-prev-clear").click(),
      "+": () => changeSpeed(+1),
      "=": () => changeSpeed(+1), // same key as + without Shift
      "-": () => changeSpeed(-1),
    };
    const action = actions[ev.key];
    if (!action) return;
    // A focused slider already moves itself with the arrow keys (and then
    // fires "input", which calls setPiece); don't move twice.
    if (ev.target.tagName === "INPUT" && ev.key.startsWith("Arrow")) return;
    ev.preventDefault(); // e.g. stop Space from scrolling the page or pressing a focused button
    action();
  });
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------
async function main() {
  wireControls();
  try {
    state.index = (await loadJSON("data/index.json")).replays;
  } catch (err) {
    const fromFile = location.protocol === "file:";
    showMessage(fromFile
      ? "Browsers don't let a page opened as a file load other files. Serve the folder instead: " +
        "<code>python -m http.server 8000 --directory site</code>, then open " +
        "<a href='http://localhost:8000'>http://localhost:8000</a>."
      : "No games found. Build them with <code>python -m scripts.build_site</code>, then reload. " +
        `(${err.message})`);
    return;
  }
  fillPicker($("pick-a"), false);
  fillPicker($("pick-b"), true);

  // Start from the URL if it names games, else the highest-scoring "best
  // checkpoint" game (sort with a comparison function: highest score first).
  const url = readURL();
  const bests = state.index.filter((e) => e.best).sort((x, y) => y.score - x.score);
  const first = bests[0] || state.index[0];
  const known = (id) => state.index.some((e) => e.id === id);
  await choose(0, known(url.a) ? url.a : first.id);
  if (url.b && known(url.b)) await choose(1, url.b);
  setPiece(url.n);
  // A shared link can name a game that this build of data/ doesn't include.
  const missing = [url.a, url.b].filter((id) => id && !known(id));
  if (missing.length) {
    showMessage(`Not in this site's data: ${missing.join(", ")}. ` +
                "Rebuild with <code>python -m scripts.build_site</code> (e.g. <code>--all</code>).");
  }
}

main();
