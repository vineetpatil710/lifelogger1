#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio → Conversation → Question Answering (Whisper + Lightweight RAG)

Usage:
  python audio_conversation_qa.py --audio "/path/to/audio_file.(wav|mp3|m4a|flac)" \
                                  --question "What did the speaker say about timelines?"

python audio_conversation_qa.py --audio "audio2.mp3" --question "explain iphone 17 pro"

What it does:
1) Transcribes the conversation from an audio file using OpenAI Whisper (offline).
2) Chunks the transcript with timestamps.
3) Retrieves the most relevant chunks for your question via semantic search.
4) Attempts extractive QA on each top chunk; if not confident, uses a generative fallback.
5) Prints the best answer and shows timestamped sources from the audio.

Dependencies (install if needed):   
  pip install openai-whisper transformers sentence-transformers torch numpy

Tip: For faster inference on supported hardware, install PyTorch with GPU support.
"""

import argparse #lets your script read command-line options like --input file.wav.
import re #“find/replace by pattern” (regular expressions).
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict #type hints to make code clearer.

import numpy as np #– fast math on arrays (vectors/matrices).
import torch #– PyTorch; runs neural nets on CPU/GPU.
import whisper #OpenAI’s speech-to-text (transcribes audio files).
from sentence_transformers import SentenceTransformer #turns sentences into vectors (embeddings) you can compare.
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# ------------------------------ Utilities ------------------------------ #

def human_time(s: float) -> str:
    """Seconds → hh:mm:ss.mmm"""
    if s is None or np.isnan(s):
        return "?:?:?.???"
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------ Data Types ------------------------------ #

@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    start: float
    end: float
    text: str
    seg_indices: List[int]


# ------------------------------ ASR (Whisper) ------------------------------ #


def transcribe_audio(
    audio_path: str,
    whisper_model: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Segment]:
    
    print(audio_path)
    print("vineet")

    """
    Transcribe audio to segments (start, end, text) using openai-whisper.
    Returns a list of Segment(start, end, text).
    """
   

    # Pick a compute device:
    # - use the provided `device` if given (e.g., "cuda", "cpu", "mps")
    # - otherwise auto-detect (e.g., prefer GPU if available)
    device = device or device_auto()

    # Load the Whisper model onto that device (sizes: tiny/base/small/medium/large-*)
    model = whisper.load_model(whisper_model, device=device)

    # Build decoding options:
    # task="transcribe" for speech->same-language text
    # fp16=True only if we're on CUDA (faster/less memory on NVIDIA GPUs)
    # language can be forced (e.g., "en"); if None, Whisper auto-detects
    options = {
        "task": "transcribe",
        "fp16": (device == "cuda"),
        "language": language
    }

    # Run transcription. We pass only non-None options to Whisper.
    result = model.transcribe(
        audio_path,
        **{k: v for k, v in options.items() if v is not None}
    

    )
   

    # Collect cleaned segments: (start time, end time, text)
    segments = []
    for seg in result.get("segments", []):
        # Normalize whitespace/punctuation quirks from ASR output
        text = normalize_space(seg.get("text", ""))
        # Only keep segments that actually have text
        if text:
            segments.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text
                )
            )
            
    full_text = " ".join(s.text for s in segments)
    # print(full_text)
    
    extract_reminder_lines(full_text)


    # print_reminders_schedules_tasks(full_text)

    return segments















import re
from typing import List, Tuple, Optional

# --- Optional: parse human dates to a tidy form if dateutil is available -----
try:
    from dateutil import parser as dtparse  # pip install python-dateutil
except Exception:
    dtparse = None  # graceful fallback; we’ll still show the time phrase text

# ----------------------------- Helpers ---------------------------------------
WS = re.compile(r"\s+")
def norm(s: str) -> str:
    return WS.sub(" ", s.strip())

# Time/Date phrase patterns (broad but practical)
TIME_PATTERNS = [
    r"\b(?:today|tomorrow|tmrw|tonight|this (?:morning|afternoon|evening|night))\b",
    r"\bnext (?:week|month|year|mon|tue(?:s)?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\b",
    r"\b(?:mon|tue(?:s)?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\b",
    r"\b(?:noon|midnight|eod|eom|eow|eoy)\b",
    r"\b(?:in|after)\s+\d+\s+(?:min(?:ute)?s?|hour(?:s)?|day(?:s)?|week(?:s)?|month(?:s)?|year(?:s)?)\b",
    r"\bby\s+(?:\d{1,2}(:\d{2})?\s?(?:am|pm)|noon|midnight|tomorrow|today|[A-Za-z]{3,9}(?:day)?)\b",
    r"\bat\s+\d{1,2}(:\d{2})?\s?(?:am|pm)\b",
    r"\b(?:\d{1,2}[:.]\d{2})\b",  # 24h or 12h without am/pm
    r"\b(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:\s+\d{2,4})?)\b",
    r"\b(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}(?:[-/\.]\d{2,4})?)\b",
]
TIME_REGEX = re.compile("|".join(TIME_PATTERNS), re.IGNORECASE)

# Common “reminder-y” cues
TRIGGERS = [
    "remind me", "reminder", "remember to", "don't forget", "do not forget", "note to self",
    "to-do", "todo", "task", "action item", "deadline", "due",
    "schedule", "appointment", "meeting", "call", "follow up", "follow-up",
    "pay", "renew", "submit", "send", "email", "buy", "pick up", "collect", "book", "arrange",
]

BULLET = re.compile(r"^\s*(?:[-*•]|-\s*\[ \])\s+", re.IGNORECASE)

# ------------------------- Core extraction ------------------------------------
def split_sentences(text: str) -> List[str]:
    # Soft sentence splitter: periods, question/exclaim marks, or line breaks
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def find_time_phrases(s: str) -> List[str]:
    matches = []
    for m in TIME_REGEX.finditer(s):
        chunk = norm(m.group(0))
        matches.append(chunk)
    # keep order & unique
    seen = set()
    uniq = []
    for x in matches:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq

def looks_like_reminder(s: str) -> bool:
    ls = s.lower()
    if BULLET.search(s):
        return True
    # has trigger OR (has a task-ish verb and a time phrase)
    has_trigger = any(t in ls for t in TRIGGERS)
    has_time = bool(TIME_REGEX.search(s))
    return has_trigger or has_time

def squeeze_task_text(s: str, time_chunks: List[str]) -> str:
    orig = " " + s + " "
    # Remove bullets
    s_ = BULLET.sub("", s).strip()

    # Anchor phrases that usually precede the actual task
    lead_patterns = [
        r"(?i)^.*?\b(?:remind me to|remember to|don'?t forget to|note to self to)\b\s*",
        r"(?i)^.*?\b(?:reminder\s*:\s*)",
        r"(?i)^\s*please\s+",
        r"(?i)^\s*can you\s+",
        r"(?i)^\s*kindly\s+",
    ]
    for pat in lead_patterns:
        s_ = re.sub(pat, "", s_, count=1)

    # If we still have “remind me/remember to/don't forget to” without the 'to', handle that:
    s_ = re.sub(r"(?i)\b(remind me|remember|don'?t forget)\b\s*", "", s_, count=1)

    # Remove obvious time prepositions when followed by a time phrase
    for ch in time_chunks:
        # include common preps near the chunk
        s_ = re.sub(rf"(?i)\b(on|at|by|before|after|this|next|coming)\s+{re.escape(ch)}", " ", s_)
        s_ = s_.replace(ch, " ")

    # If sentence starts with connectors like "to ", keep the verb phrase
    s_ = re.sub(r"(?i)^\s*(?:to\s+)", "", s_)

    # Trim filler pronouns/articles repeatedly
    s_ = re.sub(r"(?i)\b(my|the|a|an)\b\s+", " ", s_)
    s_ = norm(s_)
    # Keep it short
    if len(s_) > 100:
        s_ = s_[:97].rstrip() + "…"
    # Capitalize first letter
    if s_:
        s_ = s_[0].upper() + s_[1:]
    return s_

def tidy_when(chunks: List[str]) -> Optional[str]:
    if not chunks:
        return None
    pretty = " ".join(chunks)
    # Try to parse a single clear timestamp to a cleaner form, else keep text
    if dtparse and len(chunks) == 1:
        try:
            dt = dtparse.parse(chunks[0], fuzzy=True, dayfirst=True)  # dayfirst handles 10/12 vs 12/10
            # Show readable form without timezone assumptions
            pretty = dt.strftime("%a, %d %b %Y %I:%M %p").replace(" 12:00 AM", "")
        except Exception:
            pass
    return pretty

def extract_reminder_lines(text: str) -> List[str]:
    lines: List[str] = []
    seen = set()

    for s in split_sentences(text):
        if not looks_like_reminder(s):
            continue
        time_chunks = find_time_phrases(s)
        what = squeeze_task_text(s, time_chunks)
        when = tidy_when(time_chunks)
        if not what:  # fallback to original if extraction failed
            what = norm(s)
        short = f"{what} — {when} " if when else what
        key = short.lower()
        if key not in seen:
            seen.add(key)
            lines.append(short)

    # Also scan raw bullet lines that might not end with punctuation
    for raw in text.splitlines():
        if BULLET.search(raw):
            s = norm(BULLET.sub("", raw))
            if not s:
                continue
            time_chunks = find_time_phrases(s)
            what = squeeze_task_text(s, time_chunks)
            when = tidy_when(time_chunks)
            short = f"{what} — {when} " if when else what
            key = short.lower()
            if key not in seen:
                seen.add(key)
                lines.append(short)

    print("\n Reminder:")
    for idx, line in enumerate(lines, 1):
        print(f"{idx}. {line}")
    return lines





      



















# ------------------------------ Chunking ------------------------------ #

def chunk_segments(
    segments: List[Segment],
    max_chars: int = 1200,
    overlap_chars: int = 200
) -> List[Chunk]:
    """
    Merge consecutive segments into text windows ~max_chars with controlled overlap.
    """
    chunks: List[Chunk] = []
    buf = []
    buf_len = 0
    buf_start = None
    buf_indices = []

    def flush():
        nonlocal chunks, buf, buf_len, buf_start, buf_indices
        if not buf:
            return
        text = normalize_space(" ".join(buf))
        start = buf_start
        end = current_end if buf_indices else start
        chunks.append(Chunk(start=start, end=end, text=text, seg_indices=buf_indices.copy()))
        buf, buf_len, buf_start, buf_indices = [], 0, None, []

    current_end = 0.0
    for i, seg in enumerate(segments):
        seg_text = seg.text
        seg_len = len(seg_text)
        if buf_len == 0:
            buf_start = seg.start
        if buf_len + seg_len <= max_chars:
            buf.append(seg_text)
            buf_len += seg_len + 1
            buf_indices.append(i)
            current_end = seg.end
        else:
            flush()
            # Overlap: carry last overlap_chars chars into new buffer start
            if chunks:
                carry_text = chunks[-1].text[-overlap_chars:]
                # Re-anchor carry within new chunk without timestamps (just for text continuity)
                buf.append(carry_text)
                buf_len = len(carry_text)
                buf_start = seg.start  # new chunk timestamp starts at current seg
            else:
                buf_len = 0
                buf_start = seg.start
            buf.append(seg_text)
            buf_len += seg_len + 1
            buf_indices = [i]
            current_end = seg.end

    flush()
    return chunks


# ------------------------------ Retrieval ------------------------------ #

class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.device = device_auto() if device is None else device
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embs

    def top_k(self, query: str, chunks: List[Chunk], k: int = 5) -> List[Tuple[int, float]]:
        chunk_texts = [c.text for c in chunks]
        q = self.encode([query])  # (1, d)
        c = self.encode(chunk_texts)  # (n, d)
        sims = (c @ q.T).squeeze(-1)  # cosine sims since normalized
        top_idx = np.argsort(-sims)[:max(1, min(k, len(chunks)))]
        return [(int(i), float(sims[i])) for i in top_idx]


# ------------------------------ Readers (Extractive + Generative) ------------------------------ #

class QAReader:
    def __init__(
        self,
        extractive_model: str = "deepset/roberta-base-squad2",
        generative_model: str = "google/flan-t5-large",
        device: Optional[str] = None
    ):
        self.device = device_auto() if device is None else device

        # Extractive pipeline (fast and precise when answer span exists)
        self.qa = pipeline(
            "question-answering",
            model=extractive_model,
            device=0 if self.device == "cuda" else -1
        )

        # Generative fallback (handles synthesis across multiple chunks)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generative_model)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(generative_model)
        if self.device != "cpu":
            self.gen_model = self.gen_model.to(self.device)

    def extractive_best(
        self,
        question: str,
        candidates: List[Tuple[Chunk, float]]
    ) -> Optional[Dict]:
        """
        Try extractive QA over top chunks; return best if confident.
        """
        best = None
        for chunk, sim in candidates:
            try:
                out = self.qa(question=question, context=chunk.text)
                score = float(out.get("score", 0.0))
                answer = normalize_space(out.get("answer", ""))
                if not answer:
                    continue
                entry = {
                    "answer": answer,
                    "score": score,
                    "start_char": int(out.get("start", -1)),
                    "end_char": int(out.get("end", -1)),
                    "chunk": chunk,
                    "retrieval_sim": sim
                }
                if best is None or entry["score"] > best["score"]:
                    best = entry
            except Exception:
                continue
        # Heuristic threshold: accept only confident spans
        if best and best["score"] >= 0.35:
            return best
        return None

    def generative_synthesize(
        self,
        question: str,
        candidates: List[Tuple[Chunk, float]],
        max_context_chars: int = 3000,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        Build a compact prompt from top chunks and synthesize an answer.
        """
        # Concatenate contexts until limit
        acc = []
        used = 0
        sources = []
        for chunk, sim in candidates:
            t = f"[{human_time(chunk.start)}–{human_time(chunk.end)}] {chunk.text}"
            if used + len(t) > max_context_chars and acc:
                break
            acc.append(t)
            used += len(t)
            sources.append((chunk.start, chunk.end))
        context = "\n\n".join(acc)

        prompt = (
            "You are an expert conversation analyst. Answer the question ONLY using the context from the transcript.\n"
            "If you are not sure, provide the best supported answer from the given excerpts. Be concise and precise.\n\n"
            f"Question: {question}\n\n"
            f"Transcript excerpts:\n{context}\n\n"
            "Answer:"
        )

        inputs = self.gen_tokenizer([prompt], return_tensors="pt", truncation=True, max_length=4096)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                #  temperature=0.2,
                #  top_p=0.95
              
            )
        ans = self.gen_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        ans = ans.split("Answer:", 1)[-1].strip()
        return {"answer": ans, "score": None, "chunk": None, "sources": sources}

# ------------------------------ Orchestration (fixed, safer, and MP3-clean) ------------------------------ #
from typing import Any, List, Tuple, Optional, Dict
import os, subprocess, sys

# --- Clean MP3 → WAV to avoid mpg123 ID3 warnings ------------------------------------------

def to_clean_wav(path: str) -> str:
    """
    If input is an MP3 (or anything not already WAV), convert to 16 kHz mono WAV
    with metadata stripped. Falls back to original path if ffmpeg fails.
    """
    try:
        # If already a .wav, just return it
        if os.path.splitext(path)[1].lower() == ".wav":
            return path
        out = os.path.splitext(path)[0] + "_clean.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-map_metadata", "-1", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return out
    except Exception:
        # If conversion isn't possible, use the original (pipeline may still work)
        return path

# If your project already defines human_time, this guard prevents redefinition.
try:
    human_time  # type: ignore[name-defined]
except NameError:
    def human_time(seconds: float) -> str:
        seconds = float(seconds or 0)
        if seconds < 0:
            seconds = 0
        m, s = divmod(int(round(seconds)), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

def _has_attrs(obj: Any, *names: str) -> bool:
    return all(hasattr(obj, n) for n in names)

def map_span_to_timestamp(span_text: str, chunk: Any, segments: List[Any]) -> Tuple[float, float]:
    """
    Robustly approximate timestamps for an extractive span by locating it inside the chunk's segments.
    Falls back to the chunk window if no segment match is found.
    """
    # Fallbacks if chunk is incomplete
    ck_start = getattr(chunk, "start", 0.0)
    ck_end = getattr(chunk, "end", ck_start)

    # If the span is empty or chunk has no indices, just return the chunk window
    if not span_text or not hasattr(chunk, "seg_indices"):
        return float(ck_start), float(ck_end)

    span = span_text.lower().strip()
    span_tokens = [t for t in span.split() if t]

    best_match: Optional[Tuple[float, float]] = None
    # Iterate over candidate segments within the chunk
    for idx in getattr(chunk, "seg_indices", []):
        if not isinstance(idx, int) or idx < 0 or idx >= len(segments):
            continue
        seg = segments[idx]
        if not _has_attrs(seg, "text", "start", "end"):
            continue

        txt = str(seg.text).lower()
        # Score: exact substring OR number of leading tokens found
        if span and span in txt:
            return float(seg.start), float(seg.end)
        if span_tokens:
            hits = sum(1 for w in span_tokens[:3] if w in txt)
            if hits >= 2:  # "substantial part": at least 2 of first 3 tokens
                best_match = (float(seg.start), float(seg.end))
                break

    if best_match:
        return best_match

    # Last resort: return chunk window
    return float(ck_start), float(ck_end)


def answer_from_audio(
    audio_path: str,
    question: str,
    whisper_model: str = "small",
    language: Optional[str] = None,
    retriever_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    extractive_model: str = "deepset/roberta-base-squad2",
    generative_model: str = "google/flan-t5-large",
    top_k: int = 5
) -> Dict[str, Any]:
    # --- PRESTEP: sanitize audio to avoid mpg123 ID3 warnings -------------------
    audio_path = to_clean_wav(audio_path)

    # 1) ASR (guarded)
    try:
        segments = transcribe_audio(audio_path, whisper_model=whisper_model, language=language)
    except Exception as e:
        return {"answer": "(Transcription failed.)", "error": str(e), "sources": []}

    if not segments:
        return {"answer": "(No transcript produced.)", "sources": []}

    # 2) Chunking
    chunks = chunk_segments(segments, max_chars=1200, overlap_chars=200)
    if not chunks:
        return {"answer": "(No chunks produced from transcript.)", "sources": []}

    # 3) Retrieval
    retriever = Retriever(model_name=retriever_name)
    ranked = retriever.top_k(question, chunks, k=top_k) or []
    candidates = []
    for i, sim in ranked:
        # Defensive: ensure index is valid
        if isinstance(i, int) and 0 <= i < len(chunks):
            candidates.append((chunks[i], sim))
    if not candidates:
        # If retrieval fails, still provide a generic answer path
        candidates = [(chunks[0], 0.0)]

    # 4) Readers
    reader = QAReader(extractive_model=extractive_model, generative_model=generative_model)

    # Try extractive first (guard key access)
    ext = reader.extractive_best(question, candidates)
    if isinstance(ext, dict) and ext.get("answer"):
        ans = ext.get("answer", "")
        score = float(ext.get("score", 0.0))
        chunk = ext.get("chunk", None)

        if chunk is not None and _has_attrs(chunk, "start", "end"):
            try:
                start_ts, end_ts = map_span_to_timestamp(ans, chunk, segments)
                return {
                    "answer": ans,
                    "confidence": round(score, 4),
                    "sources": [{
                        "window": [human_time(getattr(chunk, "start", 0.0)),
                                   human_time(getattr(chunk, "end", 0.0))],
                        "precise_span": [human_time(start_ts), human_time(end_ts)]
                    }]
                }
            except Exception:
                # Fall back to window-only source if mapping fails
                return {
                    "answer": ans,
                    "confidence": round(score, 4),
                    "sources": [{
                        "window": [human_time(getattr(chunk, "start", 0.0)),
                                   human_time(getattr(chunk, "end", 0.0))]
                    }]
                }

        # If no chunk info, still return the extractive answer
        return {
            "answer": ans,
            "confidence": round(score, 4),
            "sources": []
        }

    # Fallback to generative synthesis
    gen = reader.generative_synthesize(question, candidates)
    gen_answer = (gen or {}).get("answer", "")
    gen_sources = (gen or {}).get("sources", [])

    out_sources = []
    for item in gen_sources:
        # Accept either tuple/list (start, end) or dict with 'start','end'
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            s, e = item[0], item[1]
            out_sources.append({"window": [human_time(s), human_time(e)]})
        elif isinstance(item, dict):
            s, e = item.get("start", 0.0), item.get("end", 0.0)
            out_sources.append({"window": [human_time(s), human_time(e)]})

    return {
        "answer": gen_answer,
        "confidence": None,
        "sources": out_sources or []
    }
























#!/usr/bin/env python3
"""
behavior_from_audio.py

Analyze a person's behavior from an audio file:
- Transcribe speech (OpenAI Whisper)
- Extract voice features (pitch, energy, pauses, speech rate)
- Extract text features (fillers, hedges, politeness, "we"/"I" pronouns)
- Optional text sentiment/emotions (Transformers; falls back gracefully)
- Produce an explainable behavior summary with evidence

NOT MEDICAL/CLINICAL. For conversational insight only.
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

# Whisper for ASR (requires ffmpeg installed)
import whisper

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()   # hide “unused weights” & other info-level logs


# Try to load a text sentiment/emotion pipeline (graceful fallback if unavailable)
try:
    import torch
    from transformers import pipeline
    _DEVICE = 0 if torch.cuda.is_available() else -1
    # A robust general sentiment model; you can swap to GoEmotions if you prefer fine-grained emotions
    _SENT_PIPE = pipeline("sentiment-analysis",
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                          device=_DEVICE)
except Exception:
    _SENT_PIPE = None


# --------------------------- Text helpers -------------------------------------

FILLERS = r"\b(um+|uh+|erm+|hmm+|like|you know|i mean|sort of|kinda|actually|basically)\b"
HEDGES  = r"\b(maybe|perhaps|somewhat|probably|possibly|i think|i guess|it seems|i feel)\b"
POLITE  = r"\b(please|thank you|thanks|sorry|appreciate it|could you|would you)\b"
NEG_TONE_WORDS = r"\b(angry|annoyed|frustrated|upset|disappointed|worried|anxious)\b"
POS_TONE_WORDS = r"\b(glad|happy|pleased|excited|relieved|proud|grateful)\b"

FIRST_PERSON = r"\b(i|i'd|i'll|i'm|i've|me|my|mine)\b"
SECOND_PERSON = r"\b(you|you'd|you'll|you're|you've|your|yours)\b"
WE_WORDS = r"\b(we|we'd|we'll|we're|we've|us|our|ours|together)\b"

WORD_RX = re.compile(r"[A-Za-z']+")
def count_rx(rx: str, text: str) -> int:
    return len(re.findall(rx, text, flags=re.IGNORECASE))

def word_count(text: str) -> int:
    return len(WORD_RX.findall(text))

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

# -------------------------- Data classes --------------------------------------

@dataclass
class AcousticFeatures:
    duration_s: float
    voiced_ratio: float
    num_pauses: int
    mean_pause_s: float
    rms_mean: float
    rms_std: float
    f0_mean: float
    f0_std: float
    zcr_mean: float
    speech_rate_wpm: float
    articulation_wpm: float

@dataclass
class TextFeatures:
    n_words: int
    fillers: int
    hedges: int
    polite: int
    first_person: int
    second_person: int
    we_words: int
    pos_tone_words: int
    neg_tone_words: int
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]

@dataclass
class BehaviorScores:
    confidence: int
    stress_tension: int
    empathy: int
    assertiveness: int
    politeness: int
    engagement: int

# ----------------------- Audio/ASR processing ---------------------------------

def transcribe(audio_path: str,
               whisper_model: str = "small",
               language: Optional[str] = None) -> Tuple[str, float]:
    """
    Transcribe with Whisper; return (text, duration_seconds).
    """
    model = whisper.load_model(whisper_model)
    # Whisper handles most formats via ffmpeg
    result = model.transcribe(audio_path, language=language) if language else model.transcribe(audio_path)
    text = result.get("text", "").strip()

    # Get duration robustly (soundfile first, else librosa)
    try:
        with sf.SoundFile(audio_path) as f:
            duration_s = len(f) / float(f.samplerate)
    except Exception:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration_s = librosa.get_duration(y=y, sr=sr)

    return text, float(duration_s)

def extract_acoustic(audio_path: str, target_sr: int = 16000) -> AcousticFeatures:
    """
    Extract core voice features from audio.
    """
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    duration_s = librosa.get_duration(y=y, sr=sr)

    # Energy (RMS)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))

    # Zero-crossing rate (proxy for noisiness/breathiness)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
    zcr_mean = float(np.mean(zcr))

    # Pitch (F0) via PYIN; returns NaNs in unvoiced
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=512
        )
        f0_valid = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) else 0.0
        f0_std  = float(np.std(f0_valid))  if len(f0_valid) else 0.0
        voiced_ratio = float(np.mean(~np.isnan(f0)))
    except Exception:
        # Fallback if pyin fails
        f0_mean = 0.0
        f0_std = 0.0
        voiced_ratio = 0.0

    # Pauses via energy-based segmentation
    # Keep segments where energy is above a threshold (top_db ~ 30)
    non_silent = librosa.effects.split(y, top_db=30)
    # Speech duration is sum of kept segments
   
    # Pauses are the gaps between segments
    gaps = []
    for i in range(1, len(non_silent)):
        prev_end = non_silent[i-1][1]
        curr_start = non_silent[i][0]
        gaps.append((curr_start - prev_end) / sr)
    num_pauses = len(gaps)
    mean_pause_s = float(np.mean(gaps)) if gaps else 0.0

    # Placeholder WPM (needs transcript to refine); compute with articulation rate as if words=0
    speech_rate_wpm = 0.0
    articulation_wpm = 0.0

    return AcousticFeatures(
        duration_s=duration_s,
        voiced_ratio=voiced_ratio,
        num_pauses=num_pauses,
        mean_pause_s=mean_pause_s,
        rms_mean=rms_mean,
        rms_std=rms_std,
        f0_mean=f0_mean,
        f0_std=f0_std,
        zcr_mean=zcr_mean,
        speech_rate_wpm=speech_rate_wpm,
        articulation_wpm=articulation_wpm
    )

def enrich_with_text_rates(ac: AcousticFeatures, transcript: str) -> AcousticFeatures:
    n_words = word_count(transcript)
    duration_m = max(1e-9, ac.duration_s / 60.0)
    # Speech-time minutes (exclude pauses) to get articulation rate
    speech_time_m = max(1e-9, (ac.duration_s - ac.mean_pause_s * ac.num_pauses) / 60.0)
    speech_rate_wpm = n_words / duration_m
    articulation_wpm = n_words / speech_time_m
    ac.speech_rate_wpm = float(speech_rate_wpm)
    ac.articulation_wpm = float(articulation_wpm)
    return ac

# -------------------------- Text analysis -------------------------------------

def analyze_text(text: str) -> TextFeatures:
    t = text.lower()

    n_words = word_count(t)
    fillers = count_rx(FILLERS, t)
    hedges  = count_rx(HEDGES,  t)
    polite  = count_rx(POLITE,  t)
    fp      = count_rx(FIRST_PERSON, t)
    sp      = count_rx(SECOND_PERSON, t)
    wew     = count_rx(WE_WORDS, t)
    posw    = count_rx(POS_TONE_WORDS, t)
    negw    = count_rx(NEG_TONE_WORDS, t)

    sent_label, sent_score = None, None
    if _SENT_PIPE is not None and n_words > 0:
        try:
            out = _SENT_PIPE(text[:5120])  # cap for speed
            if out and isinstance(out, list):
                sent_label = str(out[0].get("label"))
                sent_score = float(out[0].get("score"))
        except Exception:
            pass

    return TextFeatures(
        n_words=n_words,
        fillers=fillers,
        hedges=hedges,
        polite=polite,
        first_person=fp,
        second_person=sp,
        we_words=wew,
        pos_tone_words=posw,
        neg_tone_words=negw,
        sentiment_label=sent_label,
        sentiment_score=sent_score
    )

# ----------------------- Scoring & Summary ------------------------------------

def score_behavior(ac: AcousticFeatures, tx: TextFeatures) -> BehaviorScores:
    """
    Heuristic mapping from features to interpretable 0-100 scores.
    Calibrated for conversational English speech.
    """

    # Confidence: steady pitch (lower f0 std), moderate-to-strong energy, few fillers/hedges, healthy WPM
    f0_stability = 1.0 - clip01(ac.f0_std / 60.0)          # 0..1
    energy_level = clip01(ac.rms_mean / (ac.rms_mean + 0.1))
    few_fillers  = 1.0 - clip01(tx.fillers / max(1, tx.n_words / 40))  # ~1 filler per 40 words ok
    rate_ok      = 1.0 - clip01(abs(ac.speech_rate_wpm - 150) / 150)   # 150 WPM sweet spot

    confidence = int(round(100 * (0.35*f0_stability + 0.25*energy_level +
                                  0.25*few_fillers + 0.15*rate_ok)))

    # Stress/Tension: high f0 std, extreme WPM (>190 or <90), many short pauses, high ZCR
    wpm_deviation = clip01(max(0, abs(ac.speech_rate_wpm - 150) - 30) / 140)  # tolerance band
    many_pauses   = clip01(ac.num_pauses / 20.0)
    short_pauses  = clip01((0.8 - ac.mean_pause_s) / 0.8) if ac.num_pauses > 0 else 0.0
    zcr_tension   = clip01(ac.zcr_mean / 0.15)
    f0_var        = clip01(ac.f0_std / 60.0)
    stress = int(round(100 * (0.30*f0_var + 0.25*wpm_deviation + 0.20*short_pauses +
                              0.15*zcr_tension + 0.10*many_pauses)))

    # Empathy: "we" language, polite markers, positive tone words, calm tempo, softer energy variance
    we_lang   = clip01(tx.we_words / max(1, tx.n_words/50))
    politeness= clip01(tx.polite / max(1, tx.n_words/80))
    pos_tone  = clip01(tx.pos_tone_words / max(1, tx.n_words/60))
    calm_rate = 1.0 - clip01(abs(ac.speech_rate_wpm - 130) / 150)
    empathy = int(round(100 * (0.35*we_lang + 0.25*politeness + 0.20*pos_tone + 0.20*calm_rate)))

    # Assertiveness: direct language (few hedges/fillers), firm energy, slightly faster than avg, more "I"/imperatives proxy
    few_hedges = 1.0 - clip01(tx.hedges / max(1, tx.n_words/60))
    firm_energy= energy_level
    slightly_fast = clip01((ac.speech_rate_wpm - 130) / 80)  # 130..210 range
    i_usage = clip01(tx.first_person / max(1, tx.n_words/40))
    assertiveness = int(round(100 * (0.30*few_hedges + 0.25*firm_energy +
                                     0.25*slightly_fast + 0.20*i_usage)))

    # Politeness: polite terms + hedging (softeners) but penalize excessive hedging
    polite_norm = politeness
    hedging_moderate = 1.0 - abs(clip01(tx.hedges / max(1, tx.n_words/50)) - 0.3)
    politeness_score = int(round(100 * (0.65*polite_norm + 0.35*hedging_moderate)))

    # Engagement: voiced ratio, fewer/shorter pauses, energetic but not chaotic, 2nd person use
    voicing = clip01(ac.voiced_ratio)
    pause_penalty = 1.0 - clip01(ac.mean_pause_s / 2.0)
    energy_eng = energy_level
    you_focus = clip01(tx.second_person / max(1, tx.n_words/50))
    engagement = int(round(100 * (0.35*voicing + 0.25*pause_penalty + 0.25*energy_eng + 0.15*you_focus)))

    return BehaviorScores(
        confidence=confidence,
        stress_tension=stress,
        empathy=empathy,
        assertiveness=assertiveness,
        politeness=politeness_score,
        engagement=engagement
    )

def summarize_behavior(scores: BehaviorScores,
                       ac: AcousticFeatures,
                       tx: TextFeatures,
                       transcript: str) -> str:
    lines = []

    lines.append("Behavior Summary (speech + language cues)")
    lines.append("--------------------------------------------------")
    lines.append(f"Confidence: {scores.confidence}/100")
    lines.append(f"Stress/Tension: {scores.stress_tension}/100")
    lines.append(f"Empathy: {scores.empathy}/100")
    lines.append(f"Assertiveness: {scores.assertiveness}/100")
    lines.append(f"Politeness: {scores.politeness}/100")
    lines.append(f"Engagement: {scores.engagement}/100")
    lines.append("")
    lines.append("Why this read (key evidence):")
    lines.append(f"- Speech rate: {ac.speech_rate_wpm:.0f} WPM (articulation: {ac.articulation_wpm:.0f} WPM)")
    lines.append(f"- Pitch mean/std: {ac.f0_mean:.0f} Hz / {ac.f0_std:.0f} Hz")
    lines.append(f"- Energy (RMS) mean/std: {ac.rms_mean:.3f} / {ac.rms_std:.3f}")
    lines.append(f"- Voiced ratio: {ac.voiced_ratio:.2f}, Pauses: {ac.num_pauses} (avg {ac.mean_pause_s:.2f}s)")
    lines.append(f"- Text: {tx.n_words} words | fillers={tx.fillers}, hedges={tx.hedges}, polite terms={tx.polite}, "
                 f"we-words={tx.we_words}, you-words={tx.second_person}")
    if tx.sentiment_label:
        lines.append(f"- Sentiment: {tx.sentiment_label} (confidence {tx.sentiment_score:.2f})")
    lines.append("")
    # Short transcript excerpt (first ~200 chars) for context
    excerpt = re.sub(r"\s+", " ", transcript.strip())
    if excerpt:
        lines.append("Transcript (excerpt): " + excerpt[:200] + ("..." if len(excerpt) > 200 else ""))
    lines.append("")
    lines.append("Notes:")
    lines.append("- Heuristic, explainable indicators from voice and wording; not a clinical or hiring assessment.")
    lines.append("- For multi-speaker audio, consider diarization first to analyze each speaker separately.")
    return "\n".join(lines)

# ------------------------------ CLI -------------------------------------------

import sys

def parse_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyze behavior from an audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # make positional audio optional so we can print help instead of crashing
    ap.add_argument("audio", nargs="?", help="Path to audio (wav/mp3/m4a/flac/ogg/mp4)")
    ap.add_argument("--model", default="medium", help="Whisper model size (tiny/base/small/medium/large)")
    ap.add_argument("--language", default=None, help="Force language code (e.g., en); default: auto")
    # tolerate unknown args (useful in notebooks/IDE runners)
    args, _unknown = ap.parse_known_args(argv)
    return args

def main(argv=None):
    args = parse_cli(argv)

    if not args.audio:
        print("Usage: python behavior_from_audio.py <audiofile> [--model small] [--language en]")
        print("Tip: drag a file into the terminal or wrap paths with spaces in quotes.")
        return  # don't raise SystemExit(2)

    if not os.path.isfile(args.audio):
        raise SystemExit(f"File not found: {args.audio}")

    # 1) Transcribe
    transcript, _ = transcribe(args.audio, whisper_model=args.model, language=args.language)

    # 2) Acoustic features (+ speech rates using transcript)
    ac = enrich_with_text_rates(extract_acoustic(args.audio), transcript)

    # 3) Text features
    tx = analyze_text(transcript)

    # 4) Scores
    scores = score_behavior(ac, tx)

    # 5) Summary
    print(summarize_behavior(scores, ac, tx, transcript))

if __name__ == "__main__":
    main()




































# ------------------------------ CLI ------------------------------ #

def parse_args():
    # Create a command-line parser with a short description.
    p = argparse.ArgumentParser(description="Ask questions about an audio conversation.")

    # File to analyze (wav, mp3, m4a, flac, etc.)
    p.add_argument("-a", "--audio", help="Path to audio file (wav, mp3, m4a, flac, etc.)")

    # The question you want to ask about that conversation.
    p.add_argument("-q", "--question", help="Your question about the conversation.")

    # Whisper ASR (speech-to-text) model size.
    p.add_argument("--whisper_model", default="small",
                   help="Whisper model size: tiny|base|small|medium|large-v2|large-v3 (default: small)")

    # Force a specific language (skip auto-detect), e.g., 'en' for English.
    p.add_argument("--language", default=None, help="Force language code (e.g., 'en'); otherwise auto-detect.")

    # How many relevant chunks to retrieve from the transcript for answering.
    p.add_argument("--top_k", type=int, default=5, help="Top K chunks to retrieve (default: 5)")

    # Model used for extractive QA (pulls exact spans from text).
    p.add_argument("--extractive_model", default="deepset/roberta-base-squad2", help="HF model for extractive QA")

    # Model used as a fallback for generative answers (when extractive isn’t enough).
    p.add_argument("--generative_model", default="google/flan-t5-large", help="HF model for generative fallback")

    # Sentence embedding model name (for retrieval).
    p.add_argument("--retriever", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer name")

    # Parse known flags; ignore anything unknown so it doesn’t crash.
    args, _ = p.parse_known_args()

    # ---------------- Interactive fallback ----------------
    # If user launched without flags (e.g., clicking "Run"), ask for inputs.
    if not args.audio:
        args.audio = input("Path to audio file: ").strip().strip('"')
    if not args.question:
        args.question = input("Your question: ").strip()

    return args


def main():
    # Get all command-line (or prompted) arguments.
    args = parse_args()

    # Call your core pipeline (you must implement `answer_from_audio` elsewhere).
    result = answer_from_audio(
        audio_path=args.audio,
        question=args.question,
        whisper_model=args.whisper_model,
        language=args.language,
        retriever_name=args.retriever,
        extractive_model=args.extractive_model,
        generative_model=args.generative_model,
        top_k=args.top_k
    )

    # ---------------- Output formatting ----------------
    print("\n=== Answer ===")
    # Safely print the answer (or a friendly message if missing).
    print(result.get("answer", "").strip() or "(No answer.)")

    # Print confidence if provided (e.g., 0.0–1.0).
    if result.get("confidence") is not None:
        print(f"\nConfidence: {result['confidence']:.4f}")

    # Show where in the audio/transcript the answer came from.
    print("\n=== from the audio 1) The timestamp that is the source of the text answer. 2) The timestamp in the audio where the answer appears. ===")
    for i, src in enumerate(result.get("sources", []), 1):
        # Each source may have a larger 'window' and an optional more precise span.
        win = src.get("window", ["?", "?"])
        if "precise_span" in src:
            ps = src["precise_span"]
            print(f"{i}")
            print(f"{i} {win[0]} → {win[1]}   |   Span {ps[0]} → {ps[1]}")
        else:
            print(f"{i} {win[0]} → {win[1]}")
    print()


if __name__ == "__main__":
    # Run the CLI when executed directly.
    main()
