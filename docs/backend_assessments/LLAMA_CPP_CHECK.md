# OMNIAMIND — Backend Assessment: llama.cpp / llama-server

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01  
**Backend target:** llama.cpp / llama-server  
**Status:** Primary local backend assessment

---

## Purpose

This document assesses whether `llama.cpp / llama-server` is acceptable as the first real local backend for OMNIAMIND.

The local branch is now the active priority path.
The first local target has already been chosen as:

```text
llama.cpp / llama-server

This document defines:

what must be verified concretely on this runtime

which endpoint should be used first

which launch/runtime settings matter for the first OMNIAMIND proxy experiment

whether llama.cpp should be classified as ACCEPT, PROVISIONAL, or REJECT for the first local path



---

Why llama.cpp is the first local target

llama.cpp provides a lightweight local server path and exposes both a native /completion endpoint and an OpenAI-compatible /v1/chat/completions endpoint. The project documentation also shows standard local launch via ./llama-server -m your_model.gguf --port 8080. 

For the first OMNIAMIND test, this matters because the runtime can be inspected locally without provider-side API opacity. That makes it the fastest realistic path to a first direct Level 1 capture under local control. This is an inference from the documented local server behavior and from the previously fixed project constraints. 


---

Position in the ecosystem

Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT

Within this chain:

OMNIAMIND needs pre-output candidate visibility

OMNIA remains post-hoc structural measurement on emitted output

OMNIA-LIMIT remains downstream structural stop / saturation logic


The first llama.cpp check is therefore about candidate observability, not benchmark quality.


---

First endpoint choice

The first endpoint to test should be:

POST /completion

not /v1/chat/completions.

Reason

The llama.cpp server documentation explicitly describes /completion and documents the relevant candidate-probability fields there, including:

n_probs

completion_probabilities

per-token probs

streaming limitation


Specifically:

if n_probs > 0, the response includes top-N token probabilities for each generated token

in streaming mode, only content and stop are returned until the end of completion


This makes /completion the cleanest first path for OMNIAMIND Level 1 capture. 

Consequence

For the first OMNIAMIND local capture:

use /completion

use non-streaming

set n_probs > 0



---

What llama.cpp appears to expose

According to the documented /completion response:

completion_probabilities is an array with one item per generated token

each item contains:

content = selected token

probs = top-N token probabilities


the length of each probs array is n_probs 


This is exactly the kind of Level 1 candidate trace OMNIAMIND needs for first-pass proxies such as:

candidate concentration

candidate dispersion

dominance volatility

rank instability


This last sentence is an architectural inference from the documented response structure and the already defined OMNIAMIND proxy layer.


---

Critical verification targets

Check 1 — Stepwise candidate trace

Question

Does /completion return completion_probabilities across the full generated sequence?

Why it matters

Without one probability table per generated token, there is no usable Level 1 trajectory.

Expected documented support

The documentation states that completion_probabilities has length n_predict, with one item per generated token. 

What must be verified empirically

whether the actual local build returns this field

whether every generated token has a corresponding probability entry

whether the length matches the real generated output


Required result

YES

If NO, llama.cpp is not acceptable for first-pass OMNIAMIND via this endpoint.


---

Check 2 — Effective candidate depth

Question

Does n_probs yield a practically useful candidate depth per token?

Why it matters

Split and dispersion proxies become weak if the runtime only exposes a trivial number of alternatives.

Expected documented support

The server documents n_probs and says each probs array has length n_probs. 

What must be verified empirically

the maximum stable usable n_probs

whether the returned probs lists are fully populated

whether shallow truncation appears in practice


Required result

YES

A meaningful first-pass candidate depth must be obtainable.


---

Check 3 — Score semantics

Question

Are the returned values in probs usable as first-pass candidate scores?

Why it matters

OMNIAMIND needs explicit and consistent score meaning for Split and Bifurcation Pressure proxies.

Expected documented support

The /completion endpoint documents token probabilities in completion_probabilities rather than opaque internal surrogates. It also states that for temperature < 0, probabilities are still computed from logits by simple softmax without other sampler settings. 

What must be verified empirically

whether the returned scores are numerically stable

whether they remain interpretable across repeated runs

whether their behavior matches the configured decoding regime


Required result

YES


---

Check 4 — Non-streaming trace integrity

Question

Does non-streaming mode preserve the full probability table?

Why it matters

The documentation explicitly says that in streaming mode only content and stop are returned until end of completion. That makes streaming unsuitable as the first OMNIAMIND capture path. 

What must be verified empirically

that non-streaming responses do include completion_probabilities

that enabling streaming degrades the candidate trace exactly as documented


Required result

YES

for non-streaming capture viability.


---

Check 5 — Reproducibility under controlled settings

Question

Can llama.cpp be run in stable repeated conditions for candidate-trace comparison?

Why it matters

OMNIAMIND needs repeatable traces under matched settings.

Expected documented support

The server supports seed, and the /completion options include sampling controls such as temperature and sampler ordering. The general server usage also documents the global --seed option. 

What must be verified empirically

repeated-run stability under matched prompt and decoding settings

stability of candidate order and scores

whether the local environment introduces hidden variability


Required result

YES

or at minimum stable interpretability across repeated runs.


---

Check 6 — Need for deeper instrumentation

Question

Is default /completion enough for first OMNIAMIND work, or is deeper runtime instrumentation immediately required?

Why it matters

If first-pass Split / Bifurcation Pressure can already be computed from completion_probabilities, then the first experiment stays small. If not, the path must escalate toward deeper instrumentation.

Relevant documented support

The server also exposes a global --all-logits option, which indicates a deeper logits-related capability exists in the runtime. 

What must be verified empirically

whether /completion alone is sufficient

whether rawer or denser candidate access is required immediately

whether --all-logits is necessary for the first proxy layer


Required result

Preferably:

Default /completion is sufficient for first-pass Level 1

If not, classification may remain PROVISIONAL pending deeper setup.


---

Preferred launch conditions

The first llama.cpp check should use:

local llama-server

a concrete GGUF model

non-streaming

fixed seed

controlled temperature

explicit n_predict

explicit n_probs


The documentation shows local launch through:

./llama-server -m your_model.gguf --port 8080

and exposes seed as a generation/server control. 

A first minimal request should target /completion, not /v1/chat/completions.


---

Minimal first capture template

A first local capture should look like this in spirit:

{
  "prompt": "Return only one word: EVEN or ODD. Question: A box contains 7 red balls and 8 blue balls. Two balls are removed without replacement. Is the probability that both removed balls are blue greater than 1/4?",
  "n_predict": 3,
  "temperature": 0,
  "seed": 42,
  "n_probs": 20,
  "stream": false
}

This payload shape is consistent with the documented /completion endpoint and its options such as prompt, temperature, and n_probs. 


---

Acceptance logic

ACCEPT

Use ACCEPT if all of the following hold:

1. completion_probabilities is returned reliably


2. one entry exists per generated token


3. probs arrays are populated at useful depth


4. score behavior is interpretable


5. repeated runs are stable enough for first-pass proxy work


6. /completion is sufficient without immediate deeper instrumentation



PROVISIONAL

Use PROVISIONAL if:

Level 1 capture exists

but depth, score behavior, or repeatability remain partly degraded

or deeper instrumentation appears likely to be needed soon


REJECT

Use REJECT if:

full per-token candidate trace is absent

candidate depth is too weak

score semantics are unusable

reproducibility collapses under local matched settings



---

Current local branch state

OpenAI branch: unresolved / paused

Local branch: active priority path

First local backend target: llama.cpp / llama-server

First endpoint target: /completion

First mode: non-streaming with n_probs > 0



---

Minimal conclusion

llama.cpp / llama-server is the correct first local runtime for OMNIAMIND because its documented /completion endpoint already exposes the most relevant first-pass Level 1 structure:

per-token generated content

per-token top-N probabilities

local reproducibility controls

no external provider opacity 


The next real step is not more architecture.

It is empirical verification of the first local capture on /completion.

