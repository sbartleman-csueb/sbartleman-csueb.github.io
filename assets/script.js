// UI helpers
const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = new Date().getFullYear();

// ===== Mini Demo (HSV heuristics) =====
const fileInput = document.getElementById('imgInput');
const canvas = document.getElementById('preview');
const ctx = canvas ? canvas.getContext('2d', { willReadFrequently: true }) : null;
const heurLabel = document.getElementById('heurLabel');
const meanHueEl = document.getElementById('meanHue');
const meanValEl = document.getElementById('meanVal');

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)); break;
      case g: h = ((b - r) / d + 2); break;
      case b: h = ((r - g) / d + 4); break;
    }
    h /= 6;
  }
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return [h * 360, s, v];
}

function classifyHue(meanH, meanV) {
  // Very simple & tunable; works decently for bananas with diffuse light
  if (meanV < 0.25) return 'Uncertain (too dark)';
  if (meanH >= 25 && meanH <= 75 && meanV > 0.4) return 'RIPE (yellow)';
  if (meanH < 25 && meanV > 0.35) return 'OVERRIPE (brown)';
  if (meanH > 75 && meanV > 0.35) return 'UNRIPE (green)';
  return 'Uncertain';
}

async function drawAndAnalyze(img) {
  const w = 320, h = Math.round((img.height / img.width) * 320);
  canvas.width = 320; canvas.height = h;
  ctx.drawImage(img, 0, 0, w, h);

  const { data } = ctx.getImageData(0, 0, w, h);
  let sumH = 0, sumV = 0, n = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const [H, , V] = rgbToHsv(r, g, b);
    // Filter out very dark pixels and highlights
    if (V > 0.15 && V < 0.98) {
      sumH += H; sumV += V; n++;
    }
  }
  const meanH = n ? sumH / n : 0;
  const meanV = n ? sumV / n : 0;
  meanHueEl.textContent = meanH.toFixed(1) + '°';
  meanValEl.textContent = meanV.toFixed(2);
  heurLabel.textContent = classifyHue(meanH, meanV);
}

if (fileInput && canvas && ctx) {
  fileInput.addEventListener('change', e => {
    const file = e.target.files?.[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => drawAndAnalyze(img);
    img.src = URL.createObjectURL(file);
  });
}

// ===== Optional: Tiny live TF.js classifier =====
const trainBtns = document.querySelectorAll('button[data-class]');
const trainStatus = document.getElementById('trainStatus');
const modelPred = document.getElementById('modelPred');
const trainBtn = document.getElementById('trainBtn');

const classes = ['unripe', 'ripe', 'overripe'];
const samples = []; // {x: Float32Array feature, y: classIndex}

function quantizeHist(hist, bins=16) {
  // Simple RGB histogram features (concatenate 3×bins)
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const rh = new Array(bins).fill(0), gh = new Array(bins).fill(0), bh = new Array(bins).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const ri = Math.min(bins - 1, Math.floor(r / (256 / bins)));
    const gi = Math.min(bins - 1, Math.floor(g / (256 / bins)));
    const bi = Math.min(bins - 1, Math.floor(b / (256 / bins)));
    rh[ri]++; gh[gi]++; bh[bi]++;
  }
  const total = width * height;
  return Float32Array.from([...rh, ...gh, ...bh].map(v => v / total));
}

trainBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    if (!ctx) return;
    const cls = btn.dataset.class;
    const x = quantizeHist();
    const y = classes.indexOf(cls);
    samples.push({ x, y });
    trainStatus.textContent = `Collected ${samples.length} sample(s).`;
  });
});

let model;
async function buildModel(inputSize) {
  model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 24, inputShape: [inputSize], activation: 'relu' }),
      tf.layers.dense({ units: 3, activation: 'softmax' })
    ]
  });
  model.compile({ optimizer: tf.train.adam(0.02), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
}

async function trainModel() {
  if (samples.length < 6) {
    trainStatus.textContent = 'Add a few samples for each class first (≥6 total).';
    return;
  }
  const xs = tf.tensor2d(samples.map(s => Array.from(s.x)));
  const ys = tf.tensor2d(samples.map(s => {
    const one = new Array(3).fill(0);
    one[s.y] = 1;
    return one;
  }));
  if (!model) await buildModel(xs.shape[1]);
  trainStatus.textContent = 'Training…';
  await model.fit(xs, ys, { epochs: 35, batchSize: 8, shuffle: true, verbose: 0 });
  trainStatus.textContent = 'Trained. Try another image to predict.';
  xs.dispose(); ys.dispose();

  // Predict immediately on current canvas
  const feat = tf.tensor2d([Array.from(quantizeHist())]);
  const pred = model.predict(feat);
  const probs = await pred.data();
  const i = probs.indexOf(Math.max(...probs));
  modelPred.textContent = `${classes[i]} (${(probs[i]*100).toFixed(1)}%)`;
  pred.dispose(); feat.dispose();
}
if (trainBtn) trainBtn.addEventListener('click', trainModel);

// ===== Quiz =====
const quiz = document.getElementById('quizForm');
const quizScore = document.getElementById('quizScore');
const answerKey = { q1:'b', q2:'b', q3:'b', q4:'b', q5:'b', q6:'b' };

if (quiz) {
  quiz.addEventListener('submit', e => {
    e.preventDefault();
    const data = new FormData(quiz);
    let correct = 0, total = Object.keys(answerKey).length;
    for (const k of Object.keys(answerKey)) {
      if (data.get(k) === answerKey[k]) correct++;
    }
    quizScore.textContent = `Score: ${correct}/${total} — ${correct === total ? 'Perfect! 🏆' : correct >= 4 ? 'Nice!' : 'Give it another go.'}`;
  });
}
