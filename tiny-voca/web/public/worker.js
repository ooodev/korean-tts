// Bilingual front end + ONNX model, off the main thread (v2.0).
//
// Messages in: {type: "load"} and {type: "speak", id, text, speed, pitchShift, pitchRange,
//               voice?, mode?}. ``voice`` is a name from config.voices (default "KSS");
//               ``mode`` is "ko-loan" or "own" and defaults from the voice: KSS speaks
//               Mode A (ko-loan, Korean with Korean-style numerals and loanword G2P),
//               F2/M2 speak Mode B (own, native English, dizione IPA).
// Messages out: status, ready, sentence, done, error.
//
// The phonemizer is the same Python the model was trained on: kofront (the
// tiny-voca front end) drives g2pK for Korean spans and dizione for English
// spans, so a code-mixed sentence is tokenised exactly as in training.
// g2pK needs a morphological analyser; python-mecab-ko has no WASM build, so
// mecab.py (a pure-Python stand-in plus a dictionary of the corpus analyses)
// stands in for it. The model is the exported free-running graph: tokens and
// stress in, waveform out, durations predicted.

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const DIZIONE_DATA = ["lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json"];

importScripts(PYODIDE + "pyodide.js", ORT + "ort.wasm.min.js");

let pyodide, phonemize, session, config;

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Bump when the files under assets/ change, so browsers fetch them again.
const VERSION = "2";

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("tiny-voca-ko-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadPhonemizer() {
  status("?뚯꽦 遺꾩꽍 以鍮?);
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage(["numpy", "nltk"]);
  const [front, dizione, dictionary, cmudict, ...data] = await Promise.all([
    download("assets/pyfront.zip", "?뚯꽦 遺꾩꽍 以鍮?),
    download("assets/dizione.zip", "?곸뼱 ?ъ쟾 以鍮?),
    download("assets/mecab-dict.json", "?뺥깭 遺꾩꽍 ?ъ쟾 以鍮?),
    download("assets/phonemizer/cmudict.zip", "?곸뼱 ?ъ쟾 以鍮?),
    ...DIZIONE_DATA.map((name) => download("assets/phonemizer/" + name, "諛쒖쓬 ?ъ쟾 以鍮?)),
  ]);
  status("?뚯꽦 遺꾩꽍 ?ㅽ뻾");
  pyodide.FS.mkdirTree("/dizione-data");
  DIZIONE_DATA.forEach((name, index) => pyodide.FS.writeFile("/dizione-data/" + name, data[index]));
  pyodide.FS.mkdirTree("/nltk_data/corpora");
  pyodide.FS.writeFile("/nltk_data/corpora/cmudict.zip", cmudict);
  pyodide.unpackArchive(front.buffer, "zip", { extractDir: "/pyfront" });
  pyodide.unpackArchive(dizione.buffer, "zip", { extractDir: "/pyfront" });
  pyodide.FS.writeFile("/pyfront/mecab-dict.json", dictionary);
  pyodide.runPython(`
import json, os, sys
os.environ["DIZIONE_DATA"] = "/dizione-data"
os.environ["MECAB_DICT"] = "/pyfront/mecab-dict.json"
sys.path.insert(0, "/pyfront")
import nltk
nltk.data.path.insert(0, "/nltk_data")
from kofront import text as ktext

def phonemize_json(text, mode):
    # ktext.phonemize dispatches exactly like training: "ko"/"ko-loan" ->
    # phonemize_mixed (g2pK + dizione loanwords), "own" -> the dizione IPA path.
    ktext.set_frontend(mode)
    result = ktext.phonemize(text)
    return json.dumps({"ids": list(result.ids), "stress": list(result.stress),
                       "symbols": list(result.symbols), "unknown": list(result.unknown)})

phonemize_json("?뚯뒪??臾몄옣?낅땲??", "ko-loan")
`);
  phonemize = pyodide.globals.get("phonemize_json");
}

async function loadModel() {
  config = await (await fetch("assets/config.json")).json();
  const bytes = await download("assets/model.onnx", "紐⑤뜽 ?ㅼ슫濡쒕뱶");
  status("紐⑤뜽 ?ㅽ뻾");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

// Sentences: Korean ends a sentence at . ! ? ??(and the vertical bar the corpus
// uses), so split on those, keeping the mark with the sentence it closes.
function sentences(text) {
  const parts = [];
  let current = "";
  for (const piece of text.replace(/\s+/g, " ").trim().split(/(?<=[.!???)\s+/)) {
    if (!piece) continue;
    current = current ? current + " " + piece : piece;
    if (/[.!???$/.test(current) || current.length > 60) { parts.push(current); current = ""; }
  }
  if (current) parts.push(current);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

// One utterance holds at most max_tokens tokens; longer sentences are cut at
// commas, then in half.
function fit(sentence, mode) {
  const result = JSON.parse(phonemize(sentence, mode));
  if (result.ids.length <= config.max_tokens - 12) return [result];
  const pieces = sentence.split(/(?<=[,;:??)\s+/);
  if (pieces.length > 1) return pieces.flatMap((piece) => fit(piece, mode));
  const words = sentence.split(" ");
  if (words.length < 2) return [];
  const half = Math.ceil(words.length / 2);
  return [...fit(words.slice(0, half).join(" "), mode), ...fit(words.slice(half).join(" "), mode)];
}

async function synthesize(phones, voiceIndex, speed, pitchShift, pitchRange) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voiceIndex)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(pitchShift), pitch_range: scalar(pitchRange),
  });
  return { audio: output.audio.data, durations: output.durations.data };
}

// V2_SPEC 짠0: the Korean corpus voice speaks Mode A (ko-loan), the English
// corpus voices speak Mode B (own).
const defaultMode = (voice) => voice === "KSS" ? "ko-loan" : "own";

async function speak({ id, text, speed, pitchShift, pitchRange, voice, mode }) {
  const voiceName = voice || config.voices[0];
  const voiceIndex = config.voices.indexOf(voiceName);
  if (voiceIndex < 0) throw new Error(`unknown voice: ${voiceName}`);
  const speakMode = mode || defaultMode(voiceName);
  const started = performance.now();
  const list = sentences(text);
  const chunks = [];
  let phonemeTime = 0;
  let modelTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = fit(list[index], speakMode);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      mark = performance.now();
      const audio = (await synthesize(phones, voiceIndex, speed, pitchShift, pitchRange)).audio;
      modelTime += performance.now() - mark;
      chunks.push(audio);
      postMessage({ type: "sentence", id, index, total: list.length,
                    phonemes: phones.symbols.filter((s) => !s.startsWith("<")).join("") });
    }
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime, modelTime,
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadPhonemizer(), loadModel()]);
      postMessage({ type: "ready", voices: config.voices, step: config.step });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
