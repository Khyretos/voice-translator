# Recognition quality (Phase 4)

## VAD / noise detection — assessment, not a rewrite

I read through `FastVAD` in detail looking for a "better way to do it."
Honest assessment: it's already well-built for what it's doing — vectorized
spectral subtraction (single FFT per block, not per-frame), a transient/
click detector, an RMS energy gate, and an optional webrtcvad confirmation
pass, all batched as numpy array ops rather than Python loops. There isn't
an obvious architecture-level win available without changing its actual
behavior (e.g. swapping in a neural VAD like Silero, which is a much bigger
dependency + latency trade-off — say if you want to explore that
specifically). So I left it as-is and focused on tuning, below, which is
where the actual "thank you"/"subscribe" symptom was coming from.

## The Whisper hallucination fix

Two real bugs, not "Whisper is just like that":

**1. The confidence thresholds were sent but never checked.**
`whisper_no_speech_threshold`, `whisper_logprob_threshold`, and
`whisper_compression_ratio_threshold` were already exposed as settings and
sent as request params to your Whisper server — but the request asked for
`response_format: "json"`, which only returns final text, not the
per-segment confidence data those thresholds apply to. So nothing ever
actually verified the server enforced them — and plenty of minimal
OpenAI-compatible Whisper servers accept those params without fully
honoring them.

Fixed: now requests `response_format: "verbose_json"` and re-checks every
returned segment's `no_speech_prob` / `avg_logprob` / `compression_ratio`
against your thresholds client-side, dropping any segment that fails —
regardless of what the server did with them. If your Whisper host doesn't
support `verbose_json` and just returns plain JSON anyway, this falls back
to the old plain-text behavior automatically (nothing to configure).

**2. The default end-of-speech pause (80ms) was shorter than a normal breath.**
The code's own comment already said "raise this if phrases are cut off
mid-sentence... recommended range 200–600ms" — but the actual default was
80ms. That's short enough to slice a sentence into fragments on a normal
pause, and short/ambiguous clips are exactly what Whisper hallucinates
fillers on. Raised the default to 300ms. This only affects Whisper
(Vosk uses its own internal endpointer; Moonshine does its own VAD) — Vosk
and Moonshine behavior is unchanged.

If you have an existing `settings/<slug>.json` with `vad_end_silence_ms: 80`
already saved, it's left alone — I didn't want to silently override a value
you may have deliberately tuned. The new 300ms default only applies to
brand-new sessions. Bump it in the UI slider if you want the same change on
an existing session.

The existing exact-match hallucination denylist (`_WHISPER_HALLUCINATIONS`)
is left as-is as a first check, but see the follow-up section below — exact
match alone turned out not to be enough in practice.

## Round 2: "it constantly says 'thank you' on the slightest sound"

The confidence filtering and 300ms pause from round 1 weren't enough on
their own. Three more contributing causes found and fixed:

**1. The minimum-dispatch floor (150ms) was still short enough that brief
noise — a click, a cough, a breath — could qualify as a "segment" and get
sent to Whisper at all.** Very short, ambiguous clips are exactly the kind
of input Whisper hallucinates fillers on, confidence filtering or not.
Raised `_MIN_DISPATCH_MS` in `vad.py` from 150ms to 300ms, so brief blips
never reach Whisper in the first place.

**2. `whisper_condition_on_previous_text` defaulted to `True`.** This feeds
each new segment's decoding the *previous* segment's transcribed text as
context. That's meant to help with continuity in long speech, but it also
means once Whisper hallucinates "Thank you." on one ambiguous segment, the
model is biased toward continuing that pattern on the *next* one too — a
well-documented amplifier for exactly this kind of repeated-filler
hallucination loop. Defaulted to `False` for both the recognition and
translate paths. Turn it back on if you specifically want better
cross-segment continuity for long continuous speech and hallucinations
aren't a problem for your setup.

**3. The hallucination denylist really was exact-match only, and real
Whisper output varies too much for that to reliably catch it** — "Thank
you." matches, but "Thanks for watching, don't forget to subscribe!" or
"Thank you so much for watching this video everyone" don't, despite being
the same underlying hallucination. Widened `is_whisper_hallucination()`
with a conservative prefix check, split into two tiers so it can't eat real
short sentences:
- Generic openers ("thank you", "bye", "goodbye") only count for
  utterances of 4 words or fewer — real speech regularly starts this way
  and keeps going ("Thank you Bob, see you tomorrow..."), so anything
  longer is left alone.
- Phrases that are essentially never said in normal conversation ("thanks
  for watching", "like and subscribe", "see you next time") are safe to
  match over a longer span, up to 10 words.

Tested against real sentences that start the same way but keep going
("Thank you for the detailed explanation of quantum mechanics") to confirm
they're *not* caught — see `tests/test_recognizers.py`.

## If you're still seeing hallucinations after all of the above

Try, in roughly this order of impact:
1. **Raise `vad_threshold`** (in dB, e.g. from -30 to -20 or -15) if your
   environment has background noise/hum sitting close to the current
   threshold — this stops those sounds from ever being classified as
   speech in the first place, which is more effective than any amount of
   post-hoc filtering.
2. **Lower `whisper_no_speech_threshold`** (e.g. from 0.6 to 0.4) — this
   makes the client-side filter *more* aggressive about dropping segments
   Whisper itself flagged as probably-silence. Counter-intuitive: a lower
   number means more gets dropped, since the check is `no_speech_prob >
   threshold`.
3. Confirm `whisper_condition_on_previous_text` is off (see above).
4. If a specific phrase keeps recurring that isn't in the denylist, it's a
   short, easy addition to `_WHISPER_HALLUCINATIONS` or the prefix lists in
   `recognizers.py` — happy to add it if you tell me what it is.

## Live latency (Whisper)

For a live conversation tool the old Whisper pipeline had three problems:

1. **Nothing was sent until you paused.** A segment was only dispatched
   after `End-of-speech pause` of silence, so 20 s of continuous talking
   meant 20 s with no captions, then one big slow request.
2. **Old audio piled up.** Up to 3 requests ran in parallel and anything
   beyond that was dropped; on a server slower than real time results came
   back out of order, chunks went missing, and captions drifted further and
   further behind what was being said.
3. **Beam search by default** (beam 5 / best-of 5), several times slower per
   request than greedy decoding on most servers.

Now (all in the Whisper settings under **Live mode**):

- **Max segment length** (default 6 s): continuous speech is cut into
  pieces at the quietest point in the last second, so text appears while
  you're still talking.
- **Live partial captions**: roughly every 0.8 s the sentence so far is
  sent for a best-guess preview — only when the server is otherwise idle,
  and discarded if the final arrives first, so it never delays real text.
- **One ordered request at a time** (`live_whisper.py`): segments finished
  while a request is in flight are merged into the next request instead of
  queueing, so however slow the server, at most one request is waiting and
  results arrive in speaking order. If the server can't keep up at all,
  audio older than ~12 s is skipped (logged as a warning) so the captions
  jump to the present instead of lagging further.
- **Low-latency decoding** (on by default): greedy decoding, overriding the
  Beam size / Best of sliders. Turn it off to use those sliders again.
- Translation no longer holds back the recognized text: in instant subtitle
  mode the recognized line shows immediately and the translation fills in
  when it's ready. The Whisper-translate client is reused between lines so
  its connection stays open.
- Server-side, if audio processing ever falls more than ~3 s behind the
  incoming audio, the oldest blocks are dropped (logged) rather than
  processed late. The browser also stops sending when ~1 s of audio is
  already waiting on a slow connection.

The debug log shows each request as `Whisper final: 2.8s audio in 0.41s`.
If that second number is regularly larger than the first, the Whisper server
itself is slower than real time (e.g. a large model on CPU) — a smaller model
(`small`, `medium`, `large-v3-turbo`) or a GPU is the fix there.
