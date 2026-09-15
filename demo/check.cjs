'use strict';
// Dependency-free archive checks; --browser additionally requires Playwright.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const root = __dirname;
const data = JSON.parse(fs.readFileSync(path.join(root, 'samples.json'), 'utf8'));
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const expected = [
  ['video_13559|3', 'purple cylinder', 'cylinder', 'cylinder', 'cylinder'],
  ['video_13945|2', '4', '3', '3', '3'],
  ['video_13513|11', '[0]', '[]', '[]', '[]'],
  ['video_10614|11', '[0,2]', '[2,3]', '[2,3]', '[2,3]'],
  ['video_11682|13', '[0]', '[1]', '[1]', '[1]'],
  ['video_14121|11', '[0,1]', '[1]', '[1]', '[1]'],
  ['video_12373|14', '[2]', '[0,2]', '[0,2]', '[0]'],
  ['video_11961|14', '[0,1,2,3]', '[0,3]', '[0,3]', '[0,2,3]'],
  ['video_11060|9', '2', '2', '1', '1'],
  ['video_10317|1', '0', '0', '1', '0']
];
const normalize = answer => {try {const parsed = JSON.parse(answer); return Array.isArray(parsed) ? JSON.stringify(parsed) : answer;} catch {return answer;}};
assert.equal(data.samples.length, 10);
assert.equal(new Set(data.samples.map(sample => sample.id)).size, 10);
assert.equal(new Set(data.samples.map(sample => sample.videoId)).size, 10);
assert.deepEqual(data.samples.map(sample => [sample.id, ...['base', 'sft', 'grpo'].map(model => normalize(sample[model].answer)), normalize(sample.truth)]), expected);
assert.deepEqual(['base', 'sft', 'grpo'].map(model => data.fullResults[model].correct), [343, 609, 606]);
assert.deepEqual(data.paired, {corrected: 59, lost: 62, bothCorrect: 547, bothWrong: 132});
assert.equal(data.samples.filter(sample => !sample.sft.correct && sample.grpo.correct).length, 1);
assert.equal(data.samples.filter(sample => sample.sft.correct && !sample.grpo.correct).length, 1);
assert.equal(data.samples.filter(sample => sample.media.gif).length, 2);
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(root, 'data.js'), 'utf8'), context);
assert.equal(JSON.stringify(context.window.FRAME_REASON_DATA), JSON.stringify(data), 'JSON and browser data must agree.');
let mediaBytes = 0, mediaFiles = 0;
for (const sample of data.samples) {
  assert.deepEqual([sample.media.width, sample.media.height, sample.media.frames, sample.media.fps, sample.media.duration], [480, 320, 128, 25, 5.12]);
  assert.match(sample.media.originalMp4Sha256, /^[0-9a-f]{64}$/);
  for (const model of ['base', 'sft', 'grpo']) {
    const answer = sample[model];
    assert.equal(answer.correct, normalize(answer.answer) === normalize(sample.truth));
    assert.equal(typeof answer.raw, 'string');
    assert(answer.raw.includes('<answer>') && answer.raw.includes('</answer>'));
    assert.match(answer.recordId, /^[0-9a-f]{64}$/);
    assert.equal(answer.recordId, sample.base.recordId, 'Three model records must identify the same input.');
  }
  for (const key of ['video', 'poster', ...(sample.media.gif ? ['gif'] : [])]) {
    assert.match(sample.media[key], /^media\/video_\d{5}\.(?:webm|jpg|gif)$/);
    const bytes = fs.readFileSync(path.join(root, sample.media[key]));
    assert.equal(hash(bytes), sample.media.hashes[key], `${sample.id}: ${key} hash`);
    mediaBytes += bytes.length; mediaFiles++;
  }
}
assert(mediaBytes < 10_000_000, 'Keep this demonstration small.');
const serialized = JSON.stringify(data);
assert(!/(?:"[A-Z]:[\\/]|GPU-[0-9a-f-]+|sk-[A-Za-z0-9_-]{16,}|"(?:remotePath|adapterPath|framePaths)"\s*:)/i.test(serialized), 'No private archive paths, GPU IDs or credentials.');
// Optional maintainer check against the pre-export local archive.
const archiveFlag = process.argv.indexOf('--archive');
if (archiveFlag !== -1) {
  const original = JSON.parse(fs.readFileSync(path.join(process.argv[archiveFlag + 1], 'data.json'), 'utf8'));
  for (const sample of data.samples) {
    const old = original.samples.find(item => item.id === sample.id);
    assert(old, `Missing source record: ${sample.id}`);
    for (const key of ['id', 'videoId', 'questionId', 'ordinal', 'type', 'question', 'choices', 'truth', 'group']) assert.deepEqual(sample[key], old[key]);
    for (const model of ['base', 'sft', 'grpo']) for (const field of ['answer', 'raw', 'correct', 'recordId']) assert.deepEqual(sample[model][field], old[model][field]);
  }
}
console.log(`PASS: 10 videos, 30 archived answers, ${mediaFiles} media hashes, ${mediaBytes} media bytes; fixed800 results and sample selection verified.`);

async function browserCheck() {
  let chromium;
  try {({chromium} = require('playwright'));}
  catch {throw new Error('Browser checks require playwright. Install it separately and optionally set BROWSER_EXECUTABLE to an existing Chromium/Chrome/Edge executable.');}
  const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_EXECUTABLE ? {executablePath: process.env.BROWSER_EXECUTABLE} : {})});
  const page = await browser.newPage({viewport: {width: 1360, height: 1000}});
  const errors = [], network = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {if (/^https?:/.test(request.url())) network.push(request.url());});
  try {
    await page.goto(pathToFileURL(path.join(root, 'index.html')).href);
    for (const sample of data.samples) {
      await page.selectOption('#sample-select', sample.id);
      await page.waitForFunction(id => document.body.dataset.sampleId === id, sample.id);
      assert.equal(await page.textContent('#question-en'), sample.question);
      for (const model of ['base', 'sft', 'grpo']) {
        assert.equal(await page.textContent(`#${model}-answer`), sample[model].answer);
        assert.equal(await page.textContent(`#${model}-raw`), sample[model].raw);
        assert.equal(await page.textContent(`#${model}-status`), sample[model].correct ? '✓ 正确' : '× 错误');
      }
      await page.click('#truth-toggle');
      assert.equal(await page.textContent('#truth-answer'), sample.truth);
      await page.waitForFunction(() => document.getElementById('video').readyState >= 2, null, {timeout: 15000});
      await page.evaluate(async () => {const video = document.getElementById('video'); video.muted = true; await video.play();});
      await page.waitForFunction(() => document.getElementById('video').currentTime > 0.12);
      assert.deepEqual(await page.evaluate(() => {const video = document.getElementById('video'); return [video.videoWidth, video.videoHeight, Number(video.duration.toFixed(2))];}), [480, 320, 5.12]);
      if (sample.media.gif) {
        await page.click('#video-fallback');
        await page.waitForFunction(() => {const image = document.getElementById('video-gif'); return !image.hidden && image.complete && image.naturalWidth === 480;});
        await page.click('#video-fallback');
      } else assert.equal(await page.locator('#video-fallback').isVisible(), false);
    }
    for (const [type, count] of [['descriptive', 4], ['explanatory', 2], ['predictive', 2], ['counterfactual', 2], ['all', 10]]) {
      await page.selectOption('#type-filter', type);
      assert.equal(await page.locator('#sample-select option').count(), count);
    }
    await page.selectOption('#sample-select', 'video_11060|9');
    for (const width of [1360, 960, 768, 390, 320]) {
      await page.setViewportSize({width, height: 1000});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `No horizontal overflow at width ${width}`);
    }
    if (process.env.DEMO_SCREENSHOT) {await page.setViewportSize({width: 1360, height: 1100}); await page.screenshot({path: process.env.DEMO_SCREENSHOT, fullPage: true});}
    assert.deepEqual(errors, [], 'No page errors.');
    assert.deepEqual(network, [], 'Demo must not make external HTTP requests.');
    console.log('PASS: all 10 VP8 videos played; 2 GIFs loaded; all answers, raw outputs, filters and responsive widths verified in this browser.');
  } finally {await browser.close();}
}
if (process.argv.includes('--browser')) browserCheck().catch(error => {console.error(error.message); process.exitCode = 1;});
