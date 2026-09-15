(() => {
  'use strict';
  const data = window.FRAME_REASON_DATA;
  const $ = id => document.getElementById(id);
  if (!data?.samples?.length) {$('load-error').hidden = false; return;}
  const names = {descriptive:'描述', explanatory:'解释', predictive:'预测', counterfactual:'反事实'};
  // Display translations only; saved English questions and raw outputs are unchanged.
  const translations = {
    'video_11060|9': {q:'一共发生了多少次碰撞？', choices:[]},
    'video_10317|1': {q:'有多少个圆柱体在运动？', choices:[]},
    'video_13559|3': {q:'最后一个与金属立方体碰撞的物体，是什么形状？', choices:[]},
    'video_13945|2': {q:'视频结束时，有多少个静止的圆柱体？', choices:[]},
    'video_13513|11': {q:'以下哪些因素导致了蓝色物体与紫色立方体的碰撞？', choices:['金属圆柱体进入场景','紫色物体与金属圆柱体碰撞','蓝色物体与橡胶圆柱体碰撞']},
    'video_10614|11': {q:'以下哪些因素不是圆柱体与球体碰撞的原因？', choices:['金属物体与圆柱体碰撞','蓝色立方体的存在','青色物体的存在','圆柱体与青色物体碰撞']},
    'video_11682|13': {q:'接下来会发生什么？', choices:['红色立方体与金属圆柱体碰撞','棕色立方体与金属圆柱体碰撞']},
    'video_14121|11': {q:'接下来会发生哪个事件？', choices:['金属立方体与黄色橡胶立方体碰撞','灰色物体与圆柱体碰撞']},
    'video_12373|14': {q:'如果移除紫色物体，会发生哪些事件？', choices:['棕色橡胶物体与灰色球体碰撞','灰色球体与棕色金属球体碰撞','圆柱体与灰色球体碰撞','圆柱体与棕色金属球体碰撞']},
    'video_11961|14': {q:'如果没有橡胶物体，以下哪些事件不会发生？', choices:['棕色物体与蓝色球体碰撞','红色物体与立方体碰撞','立方体与棕色圆柱体碰撞','立方体与蓝色球体碰撞']}
  };
  const video = $('video');
  let shown = data.samples, current = 0, gifMode = false, generation = 0;
  const notice = message => {$('video-error').textContent = message; $('video-error').hidden = !message;};
  function showGif() {
    const sample = shown[current];
    if (!sample.media.gif) {notice('此样例未附动图。请点击“下载兼容视频 WebM”用本机播放器打开，或使用支持 VP8 的浏览器。'); return;}
    gifMode = true; video.pause(); video.hidden = true; $('video-play').hidden = true;
    $('video-gif').src = sample.media.gif; $('video-gif').hidden = false;
    $('video-fallback').textContent = '返回视频播放器';
    $('video-info').textContent = '完整视频动图转码 · 循环播放';
    notice('动图使用 256 色调色板，有色彩量化；视频内容、题目与模型回答未改变。');
  }
  function prepareVideo() {
    generation++; gifMode = false; video.pause(); video.hidden = false;
    $('video-gif').hidden = true; $('video-gif').removeAttribute('src');
    const media = shown[current].media;
    $('video-fallback').hidden = !media.gif;
    $('video-fallback').textContent = '切换动图播放';
    $('video-play').hidden = false; $('video-play').textContent = '▶ 播放视频';
    video.poster = media.poster; video.src = media.video; video.load();
    $('video-download').href = media.video; $('video-download').download = shown[current].videoId + '.webm';
    $('video-info').textContent = `${media.width} × ${media.height} · ${media.duration.toFixed(2)} 秒 · WebM / VP8`;
    notice('');
  }
  async function play() {
    if (gifMode) {const image = $('video-gif'); const src = image.src; image.removeAttribute('src'); image.src = src; return;}
    const token = generation;
    try {if (video.ended) video.currentTime = 0; await video.play();}
    catch (error) {
      if (token !== generation || error.name === 'AbortError') return;
      if (error.name === 'NotAllowedError') notice('请再次点击播放按钮，或使用动图备用播放。');
      else showGif();
    }
  }
  function describe(answer, sample) {
    if (!sample.choices.length) return ({cylinder:'圆柱体', 'purple cylinder':'紫色圆柱体', sphere:'球体', cube:'立方体'})[answer] || (/^\d+$/.test(answer) ? answer + (sample.question.includes('collisions') ? ' 次' : ' 个') : '按原文显示');
    try {
      const ids = JSON.parse(answer);
      if (!Array.isArray(ids)) return '请查看原始回答。';
      if (!ids.length) return '以上选项均不选';
      return ids.map(id => `[${id}] ` + (translations[sample.id]?.choices[id] || sample.choices.find(c => c.id === id)?.text || '不存在的选项')).join('\n');
    } catch {return '答案无法按选项编号解析，请查看原始输出。';}
  }
  function render() {
    const sample = shown[current];
    document.body.dataset.sampleId = sample.id;
    $('sample-select').value = sample.id;
    $('position').textContent = `${current + 1} / ${shown.length}`;
    $('question-type').textContent = `${names[sample.type]} / ${sample.type}`;
    $('question-zh').textContent = translations[sample.id]?.q || sample.question;
    $('question-en').textContent = sample.question;
    $('video-label').textContent = `${sample.videoId} · Q${sample.questionId}`;
    $('choices').replaceChildren();
    sample.choices.forEach(choice => {
      const li = document.createElement('li'), number = document.createElement('span'), body = document.createElement('span'), en = document.createElement('small');
      number.className = 'choice-id'; number.textContent = `[${choice.id}]`;
      body.textContent = translations[sample.id]?.choices[choice.id] || choice.text;
      en.lang = 'en'; en.textContent = choice.text;
      body.append(en); li.append(number, body); $('choices').append(li);
    });
    $('choices').hidden = !sample.choices.length;
    $('truth-panel').hidden = true; $('truth-toggle').textContent = '显示标准答案'; $('truth-toggle').setAttribute('aria-expanded', 'false');
    $('truth-answer').textContent = sample.truth; $('truth-description').textContent = describe(sample.truth, sample);
    for (const model of ['base', 'sft', 'grpo']) {
      const p = sample[model];
      $(model + '-answer').textContent = p.answer;
      $(model + '-raw').textContent = p.raw;
      $(model + '-description').textContent = describe(p.answer, sample);
      $(model + '-status').textContent = p.correct ? '✓ 正确' : '× 错误';
      $(model + '-status').className = 'status ' + (p.correct ? 'correct' : 'wrong');
    }
    document.querySelectorAll('.answer details').forEach(details => {details.open = false;});
    let note = !sample.sft.correct && sample.grpo.correct ? '此题中，GRPO 纠正了 SFT 的错误。' : sample.sft.correct && !sample.grpo.correct ? '此题中，GRPO 丢失了 SFT 已答对的结果。' : sample.sft.correct ? '此题中，SFT 与 GRPO 均匹配标准答案。' : '此题中，SFT 与 GRPO 均未匹配标准答案。';
    if (sample.id === 'video_13559|3') note += ' Base 含有正确形状，但多了颜色词而不符合答案规范，不能简单归因于看错物体。';
    $('sample-note').textContent = note + ' 个别样例不能替代完整 800 题的评测结论。';
    prepareVideo();
  }
  function filter(initialId) {
    shown = data.samples.filter(sample => $('type-filter').value === 'all' || sample.type === $('type-filter').value);
    current = typeof initialId === 'string' ? Math.max(0, shown.findIndex(sample => sample.id === initialId)) : 0;
    $('sample-select').replaceChildren();
    shown.forEach(sample => {
      const option = document.createElement('option'); option.value = sample.id;
      const suffix = !sample.sft.correct && sample.grpo.correct ? ' · GRPO 纠正' : sample.sft.correct && !sample.grpo.correct ? ' · GRPO 退步' : '';
      option.textContent = `${names[sample.type]} · ${sample.videoId} / Q${sample.questionId}${suffix}`;
      $('sample-select').append(option);
    });
    render();
  }
  $('type-filter').addEventListener('change', () => filter());
  $('sample-select').addEventListener('change', event => {current = shown.findIndex(sample => sample.id === event.target.value); render();});
  $('previous').addEventListener('click', () => {current = (current - 1 + shown.length) % shown.length; render();});
  $('next').addEventListener('click', () => {current = (current + 1) % shown.length; render();});
  $('truth-toggle').addEventListener('click', () => {const open = $('truth-panel').hidden; $('truth-panel').hidden = !open; $('truth-toggle').textContent = open ? '收起标准答案' : '显示标准答案'; $('truth-toggle').setAttribute('aria-expanded', String(open));});
  $('video-play').addEventListener('click', play);
  $('video-replay').addEventListener('click', () => {if (!gifMode) video.currentTime = 0; play();});
  $('video-fallback').addEventListener('click', () => {if (gifMode) prepareVideo(); else showGif();});
  $('video-gif').addEventListener('error', () => notice('动图加载失败，请保留完整 demo 目录，或下载兼容 WebM 视频。'));
  video.addEventListener('play', () => {$('video-play').hidden = true; notice('');});
  video.addEventListener('pause', () => {if (!gifMode) {$('video-play').textContent = '▶ 继续播放'; $('video-play').hidden = false;}});
  video.addEventListener('ended', () => {$('video-play').textContent = '↻ 再看一遍'; $('video-play').hidden = false;});
  video.addEventListener('error', () => {if (!gifMode) showGif();});
  filter('video_11060|9');
})();
