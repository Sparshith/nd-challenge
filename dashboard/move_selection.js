const strip = san => san.replace(/[+#]+$/, '');

export function pickMove(game, logits, vocabulary) {
  let id = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[id]) id = i;
  const token = vocabulary[id];
  const move = game.moves({verbose: true}).find(move => strip(move.san) === strip(token));
  return {token, move};
}
