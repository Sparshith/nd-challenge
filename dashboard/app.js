import {Chess} from 'chess.js';
import {readEntry} from './onnx_meta.js';
import {createRunner} from './runner.js';
import {setSubmissionReport} from './submission.js';
import {pickMove} from './move_selection.js';
import cases from './cases.json';
import openingsText from './openings.tsv';

const $ = id => document.getElementById(id);
const strip = san => san.replace(/[+#]+$/, '');
function makeCodec(vocabulary) {
  if (!Array.isArray(vocabulary) || vocabulary.length < 2 || vocabulary.length > 65536 || !vocabulary.every(t => typeof t === 'string')) throw new Error('The vocabulary metadata must be a JSON array of 2 to 65,536 token strings.');
  const ids = new Map(vocabulary.map((token, id) => [token, id]));
  if (ids.size !== vocabulary.length) throw new Error('Vocabulary tokens must be unique.');
  if (!ids.has('<bos>')) throw new Error('Vocabulary must include <bos>.');
  const bySan = new Map();
  vocabulary.forEach((token, id) => { const key = strip(token); if (!bySan.has(key)) bySan.set(key, []); bySan.get(key).push(id); });
  const unk = ids.get('<unk>');
  const encode = san => ids.get(san) ?? ids.get(strip(san)) ?? bySan.get(strip(san))?.[0] ?? unk;
  return {vocabulary, size: vocabulary.length, bos: ids.get('<bos>'), encode,
    history: sans => { const out = [ids.get('<bos>')]; for (const san of sans) { const id = encode(san); if (id === undefined) throw new Error(`${san} is not in this model’s vocabulary.`); out.push(id); } return out; }};
}
const BASELINE_URL = 'assets/baseline.onnx', MAX_BYTES = 10_000_000, MAX_PLIES = 300, MATCH_GAMES = 20;
const openings = openingsText.split('\n').filter(line => line && !line.startsWith('#')).slice(1).map(line => { const [, name, moves] = line.split('\t'); return {name, moves: moves.split(' ')}; });
const game = new Chess(), glyphs = {k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟︎'};
let file, sandbox, report, checking = false, generation = 0, active = false;
let codec, player = 'w', selected, promotion, thinking = false, baseline, match, results = [], mode = 'match', target = MATCH_GAMES, lastMove = null, resigned = false;
const status = (text, error = false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
const half = x => x === 0.5 ? '½' : `${Math.floor(x)}${x % 1 ? '½' : ''}`;

function reset() {
  setSubmissionReport(null);
  resigned = false; generation++; sandbox?.stop(); sandbox = null; report = null; active = false; checking = false; thinking = false; match = null; results = []; target = MATCH_GAMES;
  $('stop').hidden = true; $('result').hidden = true; $('checks').hidden = true; $('sample').hidden = false;
  $('drop').hidden = false; $('filename').hidden = true; $('file-row').hidden = true;
  $('game-panel').classList.remove('live'); $('game-status').textContent = 'Pass the check to play your model here.';
  setTab('match'); clearBoard(); renderResults();
  for (const id of ['size-check', 'load-check', 'output-check']) $(id).classList.remove('pass');
}

function selectFile(next) {
  reset(); file = next;
  $('filename').textContent = file ? `${file.name} · ${(file.size / 1e6).toFixed(2)} MB` : ''; $('filename').hidden = !file;
  $('check').hidden = true;
  if (file?.size > MAX_BYTES) status('This file is over 10 MB. Export a smaller model (try --int8).', true);
  else if (file) check();
}

function validateLogits(logits, size) {
  if (!(logits instanceof Float32Array) || logits.length !== size) throw new Error(`Expected ${size.toLocaleString()} logits, one per vocabulary token.`);
  if (!logits.every(Number.isFinite)) throw new Error('Model returned non-finite logits.');
  return logits;
}

async function check() {
  if (!file || checking) return;
  reset(); checking = true; const current = generation, input = file;
  $('check').disabled = true; $('stop').hidden = false; $('checks').hidden = false; $('sample').hidden = true;
  try {
    status('Reading the ONNX file…');
    const buffer = await input.arrayBuffer(), bytes = new Uint8Array(buffer);
    if (bytes.length > MAX_BYTES) throw new Error('This file is over 10 MB. Export a smaller model (try --int8).');
    const entry = readEntry(bytes);
    if (!entry.name) entry.name = input.name.replace(/\.onnx$/i, '');
    if (current !== generation) return;
    $('size-check').classList.add('pass');
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), b => b.toString(16).padStart(2, '0')).join('');
    if (current !== generation) return;
    const entryCodec = makeCodec(entry.vocabulary);
    if (!Number.isInteger(entry.contextLength) || entry.contextLength < 64 || entry.contextLength > 4096) throw new Error('context_length metadata must be a whole number from 64 to 4096.');
    status('Loading the model with ONNX Runtime…');
    const runner = createRunner();
    sandbox = runner;
    const start = performance.now();
    const info = await runner.call('load', {bytes});
    if (current !== generation) return;
    const loadMs = performance.now() - start;
    if (info.inputs.length !== 1) throw new Error(`The model must have one input (token ids); it has ${info.inputs.length}.`);
    $('load-check').classList.add('pass');
    const timings = [];
    for (const [index, test] of cases.entries()) {
      const tokens = entryCodec.history(test.moves);
      status(`Checking history ${index + 1}/${cases.length} · ${tokens.length} tokens…`);
      const before = performance.now();
      validateLogits(await runner.call('predict', {tokens}), entryCodec.size);
      if (current !== generation) return;
      timings.push({tokens: tokens.length, milliseconds: performance.now() - before});
    }
    codec = entryCodec;
    $('output-check').classList.add('pass');
    report = {version: 2, status: 'compatibility-passed', eligibility: 'not-reviewed', name: entry.name, file: input.name, bytes: input.size, sha256: hash, opset: entry.opset, loadMs, timings, contextLength: entry.contextLength, vocabularySize: entryCodec.size, userAgent: navigator.userAgent, checkedAt: new Date().toISOString()};
    setSubmissionReport(report);
    $('metrics').textContent = `${(input.size / 1e6).toFixed(1)} MB, loads in ${(loadMs / 1000).toFixed(1)} s, ${Math.round(timings.at(-1).milliseconds)} ms per move at 64 tokens.`;
    $('checks').hidden = true; $('sample').hidden = true; $('result').hidden = false; status('');
    $('drop').hidden = true; $('filename').hidden = true; $('file-row').hidden = false;
    $('file-name').textContent = `${entry.name} · ${(input.size / 1e6).toFixed(2)} MB`;
    active = true; $('game-panel').classList.add('live'); idleMatch(); renderResults();
  } catch (error) {
    if (current === generation) { status(error.message, true); sandbox?.stop(); sandbox = null; active = false; }
  } finally {
    if (current === generation) { checking = false; $('check').disabled = false; $('check').hidden = active; $('stop').hidden = true; }
  }
}

function renderBoard() {
  $('tab-match').disabled = $('tab-play').disabled = thinking && !match;
  $('board').replaceChildren();
  const legal = selected ? game.moves({square: selected, verbose: true}) : [];
  for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) {
    const square = String.fromCharCode(97 + (player === 'w' ? col : 7 - col)) + (player === 'w' ? 8 - row : row + 1);
    const piece = game.get(square), button = document.createElement('button');
    button.className = `square ${(col + row) % 2 ? 'dark' : 'light'}${square === selected ? ' selected' : ''}${legal.some(m => m.to === square) ? ' legal' : ''}${lastMove && [lastMove.from, lastMove.to].includes(square) ? ' last' : ''}`;
    button.dataset.square = square; button.setAttribute('aria-label', square + (piece ? ` ${piece.color === 'w' ? 'White' : 'Black'} ${piece.type}` : ''));
    if (piece) { const span = document.createElement('span'); span.className = 'piece ' + piece.color; span.textContent = glyphs[piece.type]; button.append(span); }
    button.disabled = thinking || !active || resigned || !!promotion || game.isGameOver() || game.turn() !== player;
    button.onclick = () => clickSquare(square); $('board').append(button);
  }
  $('moves').textContent = game.history().map((move, i) => (i % 2 === 0 ? `${i / 2 + 1}. ` : '') + move).join(' ');
  $('undo').disabled = thinking || !game.history().length || !active;
  $('new').disabled = thinking || !active; $('black').disabled = thinking || !active;
  $('promotion').hidden = !promotion;
}

function pushMove(move) { lastMove = game.move(move); selected = null; promotion = null; }
const tokensFor = side => side.codec.history(game.history());
function gameStatus(text) { $('game-status').textContent = game.isCheckmate() ? (game.turn() === player ? 'Model wins by checkmate.' : 'You win by checkmate.') : game.isDraw() ? 'Game drawn.' : text; }

async function reply() {
  if (resigned) return;
  if (game.isGameOver() || game.turn() === player || !active) { gameStatus('Your turn.'); return; }
  const current = generation, side = {runner: sandbox, codec, contextLength: report.contextLength};
  let tokens;
  try { tokens = tokensFor(side); } catch (error) { gameStatus(`${error.message} Take the move back.`); return; }
  thinking = true; gameStatus('Model is thinking locally…'); renderBoard();
  try {
    if (tokens.length > side.contextLength) throw new Error('This game reached the model’s context limit. Start a new game.');
    const start = performance.now(), logits = validateLogits(await sandbox.call('predict', {tokens}), codec.size);
    if (current !== generation) return;
    const {move: chosen, token} = pickMove(game, logits, side.codec.vocabulary);
    if (!chosen) {
      resigned = true;
      gameStatus(`You win. Model resigns after predicting an illegal move: ${token}.`);
      return;
    }
    pushMove(chosen); gameStatus(`Model played ${chosen.san} · ${Math.round(performance.now() - start)} ms. Your turn.`);
  } catch (error) { if (current === generation) { $('game-status').textContent = error.message; active = false; sandbox?.stop(); status('Entry stopped after a play error. Check again to restart.', true); } }
  finally { if (current === generation) { thinking = false; renderBoard(); } }
}

function clickSquare(square) {
  const options = selected ? game.moves({square: selected, verbose: true}).filter(m => m.to === square) : [];
  if (options.length) {
    if (options[0].promotion) { promotion = options; renderBoard(); return; }
    pushMove(options[0]); renderBoard(); reply();
  } else { selected = selected === square ? null : game.get(square)?.color === player ? square : null; renderBoard(); }
}

async function loadBaseline(current) {
  if (baseline) return baseline;
  $('game-status').textContent = 'Loading the baseline (9.42 MB)…';
  const bytes = new Uint8Array(await (await fetch(BASELINE_URL)).arrayBuffer()), entry = readEntry(bytes);
  const runner = createRunner();
  await runner.call('load', {bytes});
  return baseline = {runner, name: entry.name, codec: makeCodec(entry.vocabulary), contextLength: entry.contextLength};
}

const plural = (n, one, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const outcome = r => r.score === 1 ? 'win' : r.score ? 'draw' : 'loss';
function renderResults() {
  const n = results.length, score = results.reduce((a, r) => a + r.score, 0), count = x => results.filter(r => r.score === x).length;
  $('h2h-score').textContent = n ? `${half(score)} – ${half(n - score)}` : '';
  $('h2h-detail').textContent = n ? `${plural(count(1), 'win')}, ${plural(count(0.5), 'draw')}, ${plural(count(0), 'loss', 'losses')} in ${plural(n, 'game')}.` : `${MATCH_GAMES} games against the baseline: ${MATCH_GAMES / 2} openings, once with each color. Watch them on the board.`;
  $('strip').replaceChildren(...results.map((r, i) => {
    const cell = document.createElement('span');
    cell.className = outcome(r); cell.title = `Game ${i + 1}, ${r.opening}: ${outcome(r)} by ${r.reason}`;
    return cell;
  }));
  $('progress').style.width = `${Math.min(100, n / target * 100)}%`;
  const inMatch = mode === 'match';
  $('top-name').textContent = inMatch ? 'Baseline' : report?.name ?? 'Model';
  $('bottom-name').textContent = inMatch ? 'Your model' : 'You';
  $('top-score').textContent = inMatch && n ? half(n - score) : '';
  $('bottom-score').textContent = inMatch && n ? half(score) : '';
  $('match').textContent = match ? 'Pause' : !n ? 'Start match' : n < target ? 'Resume' : `Play ${MATCH_GAMES} more`;
}

function clearBoard() { player = 'w'; game.reset(); selected = null; promotion = null; lastMove = null; renderBoard(); }
function idleMatch() {
  clearBoard();
  const n = results.length;
  $('game-status').textContent = !n ? `Your model plays the baseline from ${MATCH_GAMES / 2} openings, once with each color.` : n < target ? `Paused after ${plural(n, 'game')}.` : `Match done: ${plural(n, 'game')}.`;
}

function setTab(next) {
  mode = next; $('game-panel').dataset.mode = next;
  $('tab-match').setAttribute('aria-selected', next === 'match'); $('tab-play').setAttribute('aria-selected', next === 'play');
}

function endReason() {
  return game.isCheckmate() ? 'checkmate' : game.isStalemate() ? 'stalemate' : game.isThreefoldRepetition() ? 'repetition'
    : game.isInsufficientMaterial() ? 'insufficient material' : game.isDrawByFiftyMoves() ? 'fifty-move rule'
    : game.history().length >= MAX_PLIES ? `${MAX_PLIES}-ply cap` : 'context limit';
}

function playMatch() {
  if (match) { match.stopped = true; return match.done; }
  if (results.length >= target) target += MATCH_GAMES;
  setTab('match');
  const m = match = {stopped: false};
  return m.done = runMatch(m);
}

async function runMatch(m) {
  const current = generation, live = () => !m.stopped && current === generation;
  thinking = true; renderResults(); renderBoard();
  try {
    const sides = [{runner: sandbox, codec, contextLength: report.contextLength}, await loadBaseline(current)];
    while (results.length < Math.min(target, openings.length * 2) && live()) {
      const i = results.length, opening = openings[i >> 1];
      player = i % 2 ? 'b' : 'w'; game.reset(); selected = null; promotion = null;
      for (const san of opening.moves) pushMove(game.moves({verbose: true}).find(move => strip(move.san) === san));
      renderBoard(); $('game-status').textContent = `Game ${i + 1} of ${target}. ${opening.name}.`;
      let forfeit = null, forfeitReason = 'a move outside the vocabulary';
      while (!game.isGameOver() && game.history().length < MAX_PLIES && live()) {
        const mine = game.turn() === player, side = sides[mine ? 0 : 1];
        let tokens;
        try { tokens = tokensFor(side); } catch { forfeit = mine ? 0 : 1; break; }
        if (tokens.length >= side.contextLength) break;
        const logits = validateLogits(await side.runner.call('predict', {tokens}), side.codec.size);
        if (!live()) break;
        const {move: chosen, token} = pickMove(game, logits, side.codec.vocabulary);
        if (!chosen) { forfeit = mine ? 0 : 1; forfeitReason = `illegal prediction: ${token}`; break; }
        pushMove(chosen); renderBoard();
      }
      if (!live()) break;
      results.push({score: forfeit ?? (game.isCheckmate() ? (game.turn() === player ? 0 : 1) : 0.5), reason: forfeit === null ? endReason() : forfeitReason, opening: opening.name}); renderResults();
    }
  } catch (error) { if (current === generation) $('game-status').textContent = error.message; m.failed = true; }
  finally {
    if (match === m) match = null;
    if (current === generation) {
      thinking = false; renderResults(); renderBoard();
      if (!m.failed) $('game-status').textContent = results.length < target ? `Paused after ${plural(results.length, 'game')}.` : `Match done: ${plural(results.length, 'game')}.`;
    }
  }
}

async function switchTab(next) {
  if (next === mode || (thinking && !match)) return;
  if (match) await playMatch();
  setTab(next);
  if (next === 'play') { player = 'w'; newGame(); } else idleMatch();
  renderResults();
}

function newGame() { resigned = false; $('black').textContent = player === 'w' ? 'Play as Black' : 'Play as White'; game.reset(); selected = null; promotion = null; lastMove = null; gameStatus('Your turn. Select a piece, then its destination.'); renderBoard(); reply(); }

for (const tip of document.querySelectorAll('[popover]')) {
  const button = document.querySelector(`[popovertarget="${tip.id}"]`);
  tip.addEventListener('toggle', event => {
    if (event.newState !== 'open') return;
    const box = button.getBoundingClientRect();
    tip.style.top = `${box.bottom + 8}px`;
    tip.style.left = `${Math.max(16, Math.min(box.left - 12, innerWidth - tip.offsetWidth - 16))}px`;
  });
  if (matchMedia('(hover: hover)').matches) { button.onmouseenter = () => tip.showPopover(); button.onmouseleave = () => tip.hidePopover(); }
}

$('file').onchange = event => { selectFile(event.target.files[0]); event.target.value = ''; };
for (const type of ['dragover', 'dragleave', 'drop']) document.addEventListener(type, event => {
  event.preventDefault(); document.body.classList.toggle('drag', type === 'dragover');
  if (type === 'drop' && event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]);
});
$('check').onclick = check;
$('stop').onclick = () => { reset(); $('check').hidden = !file; status('Stopped.'); };
$('match').onclick = playMatch;
$('tab-match').onclick = () => switchTab('match');
$('tab-play').onclick = () => switchTab('play');
$('new').onclick = newGame;
$('black').onclick = () => { player = player === 'w' ? 'b' : 'w'; newGame(); };
$('undo').onclick = () => { resigned = false; do { if (!game.undo()) break; } while (game.history().length && game.turn() !== player); selected = null; promotion = null; lastMove = game.history({verbose: true}).at(-1) ?? null; gameStatus('Turn taken back.'); renderBoard(); reply(); };
$('promote').onclick = () => { pushMove(promotion.find(m => m.promotion === $('promotion-piece').value)); renderBoard(); reply(); };
$('report').onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'})); const a = document.createElement('a'); a.href = url; a.download = 'entry-check.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
addEventListener('pagehide', () => { sandbox?.stop(); baseline?.runner.stop(); });
renderResults(); renderBoard();

$('agent').onclick = async () => {
  const repo = 'https://github.com/Sparshith/nanoDanya-challenge/blob/main/';
  const text = `Set up a project for the nanoDanya chess challenge: a next-move chess model exported as one ONNX file under 10 MB, no search, runs in the browser.
Read the contract first: ${repo}llms.txt and follow its "Project setup" section. Do not train or build the model yet.
Fork Sparshith/nanoDanya-challenge and work from the starter folder: ${repo}starter (the baseline model in PyTorch, export.py, vocabulary, tokenizer example, README).
Baseline to beat (1303 Elo without masking, 9.45 MB): ${new URL(BASELINE_URL, location.href).href}. The starter exports it exactly with export.py --int4.
Run export.py --int4 and check.py --games 2. When every line says OK, stop and show me the folder layout.
`;
  try { await navigator.clipboard.writeText(text); } catch { return location.assign(repo + 'llms.txt'); }
  $('agent').textContent = 'Copied'; $('agent').classList.add('done');
  setTimeout(() => { $('agent').textContent = 'Copy prompt for your agent'; $('agent').classList.remove('done'); }, 1800);
};
