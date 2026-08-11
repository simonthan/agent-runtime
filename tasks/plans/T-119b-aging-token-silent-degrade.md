---
task: T-119b
title: Aging-token silent-degrade residual — vet the T-119 stub (signed_session override vs. warm-failure telemetry vs. close)
planStatus: closed-without-code
resolution: >
  Simon-ratified 2026-08-11 (Option A: accept as documented limitation). No agent-runtime change
  owed. signed_session override stays rejected (converts bounded-latency into candidate
  total-outage); warm-failure-streak telemetry is mis-targeted (msal swallows the refresh error
  in the real residual, warm returns True silently). Re-open only via Option C — a real observed
  AAD-degraded-aging incident in the field. Tracked in tbp status.md under T-119b (umbrella owner).
impact: infrastructure
blockedBy: []
---

# T-119b — Aging-token silent-degrade residual

**Outcome of planning: `needs-decision`, recommending CLOSE-WITHOUT-CODE (ratifiable by Simon).**

Every premise in the stub was re-verified against installed vendor source and agent-runtime HEAD
(`main`, `v0.21.3`, clean tree). The core finding is not a coding gap — it is that **both fixes the
stub named are unbuildable-as-described or already-rejected**, and the residual is a bounded,
no-worse-than-baseline latency effect confined to an AAD outage window. This document is the vetting
record, not an implementation plan. There is one genuine decision for Simon (§Decision).

Repo: `agent-runtime`, branch `main` (ships off `main`, NOT master). No tbp code change is implied by
any option here.

---

## The stub (as filed, tbp `status.md` order 6j)

> **Aging-token silent-degrade residual:** when AAD is down but a valid-but-aging token is cached,
> MSAL swallows the refresh error (`msal/application.py:1721-1727`) → the T-119 warm returns True and
> the on-loop sync call repeats the failing network attempt each send. No worse than pre-T-119;
> pre-warm cannot fix it — needs `signed_session` override (rejected on risk in T-119, recorded
> not-on-merit) or warm-failure-streak telemetry.

---

## Anchor verification (every claim re-read at HEAD)

| Claim | Status |
|---|---|
| `msal/application.py:1721-1727` swallows the refresh error and returns the cached token | **Correct.** Read at `agent-runtime/.venv/.../msal/application.py`. `if (result and "error" not in result) or (not access_token_from_cache): return result` — so when the refresh `result` HAS an error **and** a cached AT exists, control falls through; `except http_exceptions:` re-raises **only** `if not access_token_from_cache`; final line `return access_token_from_cache`. This is msal's `_acquire_token_silent_with_error`. |
| The warm returns `True` in the residual (does not raise) | **Correct — and this is the load-bearing fact.** Traced the full chain (below). `get_access_token` returns the cached token *string*; no exception; `warm_connector_token` returns `True`. |
| "No worse than pre-T-119" | **Correct.** The on-loop `signed_session` mint existed before T-119 and is unchanged; T-119 only added an off-loop warm ahead of it. In the residual the loop-block is identical to pre-T-119; only total turn latency grows (warm pays a doomed timeout first, then the on-loop call pays it again). |
| `signed_session` override "rejected on risk, not on merit" in T-119 | **Correct.** T-119 plan Design §1, "Rejected (but the strongest alternative)" — quoted verbatim in §1 below. |
| agent-runtime version / baseline | `v0.21.3` (`pyproject.toml` + `__init__.py` both `0.21.3`), CHANGELOG top heading `## v0.21.3 — 2026-08-09`. T-119c (sync `TokenApiClient` sign-in-resource off-loop) shipped here. |

### The token path, traced end to end (why the warm cannot see the degrade)

1. `TeamsAdapter.warm_connector_token` → `await asyncio.to_thread(self._credentials.get_access_token)`
   (T-119, `adapter.py`).
2. `self._credentials` is `BoundedAppCredentials` (`src/agent_runtime/transport/teams/_msal.py:78-86`).
   Its `get_access_token` does double-checked-lock construction then `return super().get_access_token(...)`.
3. `super()` is botframework `MicrosoftAppCredentials.get_access_token`
   (`.venv/.../botframework/connector/auth/microsoft_app_credentials.py`):
   ```python
   auth_token = self.__get_msal_app().acquire_token_silent(scopes, account=None)
   if not auth_token:
       auth_token = self.__get_msal_app().acquire_token_for_client(scopes=scopes)
   if "access_token" in auth_token:
       return auth_token["access_token"]          # <-- BARE STRING; result dict discarded
   ...
   raise PermissionError(f"Failed to get access token with error: {error}, ...")
   ```
4. In the aging-outage residual: botframework calls `acquire_token_silent(scopes, account=None)`,
   which is a **backward-compat NO-OP that returns `None` at `application.py:1491-1492`** for
   `account=None`. botframework therefore falls through to `acquire_token_for_client(scopes=scopes)`
   (the client-credentials flow), whose `_acquire_token_silent_with_error` reaches the swallow at
   `application.py:1721-1727`, swallows the doomed-refresh http error, and returns the **cached** app
   token dict (has `access_token`, no `error`). So step 3 takes the `return auth_token["access_token"]`
   branch. **The warm gets a valid string, raises nothing, returns `True`.** (The entry point is the
   `for_client` fall-through, not `acquire_token_silent` — but the swallow function
   `_acquire_token_silent_with_error` and the net outcome are identical either way; verified by Round 1
   critic.)

The consequence for observability: the only fields that distinguish "served fresh from the IdP" from
"served aging from cache after a *failed* refresh" are msal's `token_source` / `refresh_on`, and
`get_access_token` **discards the dict and returns a bare string**. The warm layer therefore has no
signal to key telemetry on — see §2.

---

## §1 — `signed_session` override: keep it REJECTED (no new information)

T-119 Design §1 recorded this verbatim (re-read at plan time):

> **Rejected (but the strongest alternative) — override `signed_session` on `BoundedAppCredentials`.**
> … genuinely more complete than the turn-boundary warm … the **only** shape that closes the
> aging-outage residual … **Rejected for this task on risk, not on merit:** it means owning our own
> expiry bookkeeping (`get_access_token` returns a bare string, so we would have to read
> `acquire_token_silent`'s dict directly), holding a raw bearer token on the object, and firing a
> background refresh from a synchronous method that may run with no running loop. Every one of those
> failure modes ends in "the wrong Authorization header goes out", i.e. 401s on every outbound
> activity — a total-outage failure mode traded for a latency one.

**Verdict: still rejected. Nothing has changed since T-119 to move it off the risk column.**

- The trade is unchanged: it converts a **bounded latency** effect (loop-block ≤ T-115m's ~60 s
  ceiling, confined to an AAD outage window) into a candidate **total-outage** effect (a stale/blank
  bearer header → 401 on *every* send). That is strictly the wrong direction for a residual whose
  worst case today is "the turn is slow."
- The residual it would close is **latency-only and no-worse-than-baseline**. Spending the platform's
  highest-risk change (hand-rolled token/expiry bookkeeping in the outbound-auth hot path) to shave
  latency off a rare outage window fails the cost/benefit test on its face.
- T-119's own exit criterion for reviving it was explicit: *"if the Mechanism 2 residual ever shows
  up in telemetry, this is the follow-up to file, and it should be sized M with its own live smoke."*
  It has **not** shown up in telemetry (there is none for it — see §2). Reviving it now would be
  building the risky option *speculatively*, which the T-119 author explicitly deferred.

If Simon ever wants this closed on merit, the correct trigger is a real, observed AAD-degraded-aging
incident in production, and the correct size is **M with a live smoke** — not this S residual.

---

## §2 — "Warm-failure-streak telemetry": the stub's OTHER option is MIS-TARGETED

This is the decisive finding of the vetting. The stub offers "warm-failure-streak telemetry" as the
low-risk buildable alternative. **It cannot observe the residual it is named for**, because in the
residual the warm does not fail. Three scenarios, kept distinct:

| Scenario | msal behaviour | `get_access_token` | `warm_connector_token` | Already observable? |
|---|---|---|---|---|
| **A. AAD healthy** | refresh/serve succeeds | returns fresh string | `True` | n/a (nominal) |
| **B. AAD down, token EXPIRED (no usable cache)** | `except http_exceptions: raise` (`:1725-1726`) | raises `PermissionError`/timeout | catches, **`logger.warning(exc_info=True)`, returns `False`** | **YES — already logs today** (T-119 C3) |
| **C. AAD down, token AGING-but-valid (the residual)** | swallows error, returns cached AT (`:1727`) | returns cached string | **`True`, logs nothing** | **NO** |

- A "warm-failure-streak" counter fires on **Scenario B**, which T-119 already surfaces with a
  `logger.warning`. Adding a streak counter on top buys a marginal "transient vs. sustained" signal
  for a case that is *not the residual* — and worse, it would be **named for a residual it stays
  silent on** (Scenario C), which is actively misleading.
- To observe **Scenario C** you must read msal's internal result dict (`token_source` /
  `refresh_on`) — the warm only has a bare string. Reaching for that dict is exactly the
  bookkeeping T-119 §1 rejected on risk. **So "observe the real residual cheaply" is not on the
  table**: the cheap telemetry watches the wrong scenario, and the telemetry that would watch the
  right scenario carries the §1 risk.

### The one genuinely low-risk, correctly-targeted signal that DOES exist

There is exactly one observable that keys on Scenario C without touching any token internals:
**warm duration.** In Scenario C the warm's `get_access_token` performs a *doomed* network refresh
(msal proactively refreshes once `refresh_on` elapses) that runs to the timeout ceiling before
falling back to cache — so the warm **succeeds but is slow** (seconds), whereas a healthy refresh
POST is sub-second. A timer around the `to_thread` call that logs at WARNING when the warm returns
`True` but exceeded, say, 3 s, is:

- low-risk (a stopwatch; no behaviour change, no token internals, no new dependency);
- correctly-targeted (fires on Scenario C's slow-fallback);
- ~6 lines in `warm_connector_token`, one test.

**But its marginal value is low, and this document recommends against building it now:**

1. It is **noisy for the right reasons**: msal does a proactive refresh ~twice an hour even when AAD
   is *healthy*; any transient slow POST (or a cold start's discovery GET) trips a duration
   threshold too, so the signal is "the refresh path was slow," not "AAD is degraded." Separating the
   two needs the very `token_source` read we are avoiding.
2. During any **real** AAD outage, Scenario B (`logger.warning`, already shipped) fires as soon as
   *any* token expires — so an operator already gets a loud signal from the same incident. The
   Scenario-C-only slice (aging window, ≤ ~25 min, token not yet expired) is a narrow sliver that a
   real outage quickly overtakes into Scenario B.
3. The condition it reports is **no-worse-than-baseline latency with correct output**. Per Simon's
   deferral-tripwire policy, a trip-wire is worth adding when it "fires as DATA before as a
   complaint" — here the complaint (elevated turn latency during an AAD outage) is already
   accompanied by Scenario B warnings, so the additional wire mostly adds noise rather than *earlier*
   data.

It is offered as the **only** buildable option, honestly scoped, so Simon can choose it if he wants
the observability anyway. It is not recommended.

---

## §3 — Why "close-without-code" is the evidence-supported verdict

1. **The residual is real but benign.** Bounded latency (≤ T-115m ceiling), confined to the
   AAD-degraded aging window, correct output throughout, **no worse than pre-T-119**. It is not a
   correctness or availability defect.
2. **Both stub-named fixes are off the table on their own terms.** `signed_session` override stays
   rejected on risk with no new information (§1); "warm-failure-streak telemetry" watches the wrong
   scenario and the correctly-targeted variant carries the §1 risk (§2).
3. **The only low-risk buildable signal (warm-duration) adds little** and is recommended against (§2).
4. **Precedent.** T-120 and T-121c (and T-123's `needs-decision`) establish that a vetted
   close/ratify verdict is a legitimate output of planning when the evidence does not support a
   build. Manufacturing the risky `signed_session` build to "have something to ship" is explicitly
   the anti-pattern.

**Recommendation: CLOSE-WITHOUT-CODE.** Record the residual as a known, accepted limitation of the
T-119 warm (it already is — T-119 CHANGELOG and Design §2 both state it), and re-open only on a real
observed production incident, at size M with a live smoke, via the `signed_session` route.

---

## Decision (for Simon)

**Ratify one:**

- **(A) Close-without-code (recommended).** Mark T-119b closed as an accepted, documented limitation.
  No agent-runtime release, no version bump, no tbp pin change, no paired tbp task.
- **(B) Build the warm-duration telemetry** (the only low-risk option; recommended *against*). If
  chosen, it is a genuine — if marginal — S build: ~6 lines in `warm_connector_token` + one test in
  `tests/transport/teams/test_adapter.py`; a patch version bump (two sites + CHANGELOG + `uv.lock`
  regen per the standard agent-runtime release dance) and a **routine tbp pin bump only** (no tbp
  code, mirroring T-119/T-119c). This document would then be re-planned to `ready-to-iav` with the
  full task/test/verification detail; it is deliberately NOT specified to that level here because the
  recommendation is (A).
- **(C) Revive `signed_session` override.** Only on a real observed AAD-degraded-aging incident;
  size M with its own live smoke; not an S residual. Not recommended now.

No option here requires touching tbp `status.md` — the tbp orchestrator records the verdict.

---

## Review Notes

Self-review (Phase 5.4): no data-persistence, dependency-manifest, migration, or signature changes in
the recommended path (close). LLM-in-the-loop checklist: the residual concerns MSAL token acquisition,
not model calls — no prompt-injection / cost surface. The one buildable alternative (warm-duration
telemetry) was traced for the "warm never raises" contract and confirmed additive.

**Round 1: Opus staff-engineer sign-off (foreground, delivered). 1 finding: 0 CRITICAL/HIGH/MEDIUM,
1 LOW.** Reviewer independently traced the full chain (`BoundedAppCredentials.get_access_token` →
botframework `get_access_token` → msal `acquire_token_silent` @1467 / `acquire_token_for_client`
@2494 / shared `_acquire_token_silent_with_error` swallow @1721-1727) and stress-tested all four
central claims.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | LOW | Prose in "token path traced" step 4 attributed the swallow to `acquire_token_silent`; in fact `acquire_token_silent(account=None)` returns `None` at `application.py:1491-1492` (documented no-op), so the swallow is reached via the `acquire_token_for_client` fall-through. Net outcome (cached string, warm returns `True`) is identical, and the anchor table already named `_acquire_token_silent_with_error` correctly. Reviewer noted this *reinforces* the verdict — the Q4 "could it return None and behave differently?" path leads to the same swallow, not to a raise. | **Applied** — step 4 corrected to the `for_client` fall-through with the `:1491-1492` no-op cited. |

Reviewer's other conclusions (recorded, no plan change): Q1 warm-cannot-observe-Scenario-C
**confirmed robustly** (both the transport-exception and http-error-response sub-cases return the
cached dict without raising); Q2 §2 enumeration of low-risk signals is **complete** (no cheaper
correctly-targeted option exists — a read-only `refresh_on` check cannot distinguish a *failed* aging
refresh from a *healthy* not-yet-refreshed one); Q3 **no correctness/availability upgrade** (output is
a valid ≥5-min cached token, no 401; doubled Scenario-C turn latency stays far under the T-080 600 s
deadline). One LOW nuance noted but not a defect: in Scenario C the warm newly occupies a shared
default-executor worker for the doomed-refresh duration each turn (pressure that was purely on-loop
pre-T-119) — bounded to the ≤25-min aging window, strictly better than starving the single loop, and
already disclosed generally in T-119 plan §2.

**Verdict after Round 1: unchanged — close-without-code (Simon ratifies; §Decision).** No
CRITICAL/HIGH/MEDIUM finding; the only correction was a LOW prose slip that leaves the conclusion
intact. Rounds 2 (Gemini) and 3 were not run: there is no code to review, the highest-value senior
(Opus) round was run and delivered, and the output is a close/ratify verdict rather than a
`ready-to-iav` build.
