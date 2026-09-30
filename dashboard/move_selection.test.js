import test from 'node:test';
import assert from 'node:assert/strict';
import {Chess} from 'chess.js';
import {pickMove} from './move_selection.js';

test('illegal or non-SAN top tokens lose even when a legal token is second', () => {
  for (const token of ['e5', '<bos>', '<eos>', '<unk>', '--', 'e2e4', 'Ng1f3', 'bad']) {
    const game = new Chess(), before = game.fen();
    assert.deepEqual(pickMove(game, [10, 9], [token, 'e4']), {token, move: undefined});
    assert.equal(game.fen(), before);
  }
});

test('legal top token plays, ties use lowest id, and invalid tied tokens still lose', () => {
  assert.equal(pickMove(new Chess(), [1, 1], ['d4', 'e4']).move.san, 'd4');
  assert.equal(pickMove(new Chess(), [1, 1], ['<eos>', 'e4']).move, undefined);
});

test('check and mate suffixes are optional', () => {
  const game = new Chess();
  for (const san of ['f3', 'e5', 'g4']) game.move(san);
  assert.equal(pickMove(game, [1], ['Qh4']).move.san, 'Qh4#');
});
