'use strict';
// Export an existing local archive; never downloads data or launches inference.
// Usage: node demo/export-archive.cjs /path/to/video-threeway [--include-media]
// Only use --include-media after checking permission to redistribute the dataset.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const source = process.argv[2];
if (!source || source.startsWith('--')) throw new Error('Pass the local video-threeway archive directory.');
const archive = JSON.parse(fs.readFileSync(path.join(source, 'data.json'), 'utf8'));
const sourceMedia = JSON.parse(fs.readFileSync(path.join(source, 'media-compatibility.json'), 'utf8'));
const includeMedia = process.argv.includes('--include-media');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const gifIds = new Set(['video_11060', 'video_10317']);
if (archive.samples.length !== 10) throw new Error('Expected exactly 10 archived examples.');
const data = {
  schemaVersion: 1,
  mode: 'archived-model-outputs-not-live-inference',
  dataset: {name: 'CLEVRER', url: 'https://clevrer.csail.mit.edu/', license: 'CC0', licenseSource: 'https://data.csail.mit.edu/clevrer/README.txt', split: 'custom fixed 800-question subset of the official validation split'},
  models: {
    family: archive.models.family,
    revision: archive.models.revision,
    base: {label: 'Base', step: null},
    sft: {label: 'SFT700', step: 700},
    grpo: {label: 'GRPO3000', step: 3000},
    input: '8 uniformly sampled frames per question',
    seed: 42
  },
  fullResults: archive.fullResults,
  paired: {corrected: 59, lost: 62, bothCorrect: 547, bothWrong: 132},
  selection: 'First two Base/SFT-disagreement examples of each question type in original evaluation order, plus the earliest GRPO correction and earliest GRPO loss. These 10 examples are illustrative, not representative.',
  limitations: [
    'Historical archived predictions, not live model inference.',
    'One seed and a custom evaluation subset, not the official hidden-test leaderboard.',
    'Chinese question/choice translations are display-only; original model prompts were English.',
    'Answer correctness is the archived strict answer score, not a judgment of explanation plausibility.',
    'Symbolic traces were annotation-matched, not executed in a physical simulator.',
    'The player shows the full original clip; the models received eight sampled frames.',
    'The public demo does not contain checkpoints or the original eight model-input JPEGs.'
  ],
  samples: archive.samples.map(s => {
    const sample = {};
    for (const field of ['id', 'videoId', 'questionId', 'ordinal', 'type', 'question', 'choices', 'truth', 'group']) sample[field] = s[field];
    for (const model of ['base', 'sft', 'grpo']) {
      const p = s[model];
      if (typeof p.raw !== 'string' || typeof p.correct !== 'boolean') throw new Error('Missing authentic answer.');
      sample[model] = {answer: p.answer, raw: p.raw, correct: p.correct, recordId: p.recordId};
    }
    const m = sourceMedia[s.videoId];
    sample.media = {
      video: `media/${s.videoId}.webm`,
      poster: `media/${s.videoId}.jpg`,
      width: m.width, height: m.height, fps: m.fps, frames: m.frames, duration: m.duration,
      codec: 'VP8', originalMp4Sha256: m.sourceOriginalSha256,
      hashes: {video: m.hashes.webm, poster: m.hashes.poster},
      derivation: 'Full-clip CPU re-encoding; no trimming or frame-rate reduction. Poster: frame 0. GIF, where provided, uses a lossy 256-color palette.'
    };
    if (gifIds.has(s.videoId)) {sample.media.gif = `media/${s.videoId}.gif`; sample.media.hashes.gif = m.hashes.gif;}
    if (includeMedia) {
      fs.mkdirSync(path.join(__dirname, 'media'), {recursive: true});
      for (const [publicKey, localKey] of [['video', 'webm'], ['poster', 'poster'], ...(gifIds.has(s.videoId) ? [['gif', 'gif']] : [])]) {
        const content = fs.readFileSync(path.join(source, m[localKey]));
        if (hash(content) !== sample.media.hashes[publicKey]) throw new Error(`Media hash mismatch: ${s.videoId}/${publicKey}`);
        fs.writeFileSync(path.join(__dirname, sample.media[publicKey]), content);
      }
    }
    return sample;
  })
};
const json = JSON.stringify(data, null, 2) + '\n';
// Allowlist above is the primary privacy control; this check catches regressions.
if (/(?:"[A-Z]:[\\/]|GPU-[0-9a-f-]+|sk-[A-Za-z0-9_-]{16,}|"(?:remotePath|adapterPath|framePaths)"\s*:)/i.test(json)) throw new Error('Private data found in public export.');
fs.writeFileSync(path.join(__dirname, 'samples.json'), json);
fs.writeFileSync(path.join(__dirname, 'data.js'), 'window.FRAME_REASON_DATA = ' + json.trimEnd() + ';\n');
console.log(JSON.stringify({samples: data.samples.length, answers: data.samples.length * 3, mediaIncluded: includeMedia}));
