"""
WoRMS name resolver (preprocessing).

Turns whatever the user/agent supplied (scientific names, misspellings, common
names) into validated AphiaIDs *before* any data tool runs. This is the WoRMS
equivalent of GBIF's parameter-resolution step, but narrower: in WoRMS almost
everything hangs off the AphiaID, so resolution is the whole job.

Resolution strategy (every branch grounded in verified WoRMS behavior):
  1. One batch `AphiaRecordsByMatchNames` call (fuzzy, returns valid_AphiaID).
  2. Per input name, walk its candidate list:
       - 0 candidates   -> vernacular fallback (AphiaRecordsByVernacular)
       - still nothing  -> unresolved
       - exactly 1      -> resolve (synonym pivot handled by ResolvedTaxon)
       - 2+ candidates  -> heuristic pick; if too close, mark ambiguous
  3. Emit a ResolutionResult (resolved taxa + unresolved names).

The resolver does no networking of its own; it reuses the WoRMS client from
worms_api.py. It is synchronous at the HTTP layer (the WoRMS client uses
requests/cloudscraper); callers run it in an executor, as the agent already
does for other blocking calls.
"""

from typing import Optional

from src.worms_api import WoRMS, MatchNamesParams, VernacularSearchParams
from src.resolution import (
    ResolvedTaxon,
    ResolutionResult,
    ResolutionSource,
    TaxonStatus,
)


# When the top two candidates score this close, we treat the match as ambiguous
# rather than silently picking. (Scores are integers; a gap of 0 means a tie.)
_AMBIGUITY_GAP = 1


class WoRMSResolver:
    """Resolves input names to validated AphiaIDs using WoRMS endpoints."""

    def __init__(self, worms: Optional[WoRMS] = None, marine_only: bool = True):
        self.worms = worms or WoRMS()
        self.marine_only = marine_only

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def resolve(self, names: list[str]) -> ResolutionResult:
        """Resolve a batch of input names. Pure orchestration over WoRMS calls.

        This is blocking (HTTP). Callers should run it in an executor.
        """
        cleaned = [n.strip() for n in names if n and n.strip()]
        if not cleaned:
            return ResolutionResult(resolved=[], unresolved=[])

        match_lists = self._match_names(cleaned)

        resolved: list[ResolvedTaxon] = []
        unresolved: list[str] = []

        for input_name, candidates in zip(cleaned, match_lists):
            taxon = self._resolve_one(input_name, candidates)
            if taxon is not None:
                resolved.append(taxon)
            else:
                unresolved.append(input_name)

        return ResolutionResult(resolved=resolved, unresolved=unresolved)

    # ------------------------------------------------------------------ #
    # WoRMS calls
    # ------------------------------------------------------------------ #

    def _match_names(self, names: list[str]) -> list[list[dict]]:
        """Batch MatchNames call. Returns one candidate list per input name,
        in the same order. On any failure, returns empty candidate lists so the
        per-name logic can fall through to the vernacular fallback."""
        try:
            params = MatchNamesParams(scientific_names=names, marine_only=self.marine_only)
            url = self.worms.build_match_names_url(params)
            raw = self.worms.execute_request(url)
        except Exception:
            return [[] for _ in names]

        # MatchNames returns a list-of-lists: one inner list per input name.
        if not isinstance(raw, list):
            return [[] for _ in names]

        # Normalize length/shape defensively — WoRMS should preserve order and
        # arity, but we never want a zip mismatch to silently misalign names.
        normalized: list[list[dict]] = []
        for i in range(len(names)):
            if i < len(raw):
                slot = raw[i]
                if isinstance(slot, list):
                    normalized.append([c for c in slot if isinstance(c, dict)])
                elif isinstance(slot, dict):
                    normalized.append([slot])
                else:
                    normalized.append([])
            else:
                normalized.append([])
        return normalized

    def _vernacular_lookup(self, common_name: str) -> list[dict]:
        """Common-name fallback via AphiaRecordsByVernacular. MatchNames does NOT
        resolve vernaculars (verified: 'killer whale' -> 204), so this is the only
        path that turns a common name into a record.

        Two-stage, mirroring our scientific-name strategy: try an EXACT vernacular
        match first, fall back to fuzzy `like` only if exact found nothing.
        Verified: 'killer whale' with like=true returns 3 species (Feresa/Orcinus/
        Pseudorca, all match_type 'like'); with like=false it returns exactly
        Orcinus orca (match_type 'exact'). Exact-first makes the common case
        correct; fuzzy is the safety net for partial/misspelled common names."""
        exact = self._vernacular_request(common_name, like=False)
        if exact:
            return exact
        return self._vernacular_request(common_name, like=True)

    def _vernacular_request(self, common_name: str, like: bool) -> list[dict]:
        """Single AphiaRecordsByVernacular call at the given match strictness."""
        try:
            params = VernacularSearchParams(vernacular_name=common_name, like=like)
            url = self.worms.build_vernacular_search_url(params)
            raw = self.worms.execute_request(url)
        except Exception:
            return []

        if isinstance(raw, list):
            return [c for c in raw if isinstance(c, dict)]
        if isinstance(raw, dict):
            return [raw]
        return []

    # ------------------------------------------------------------------ #
    # Per-name resolution
    # ------------------------------------------------------------------ #

    def _resolve_one(self, input_name: str, candidates: list[dict]) -> Optional[ResolvedTaxon]:
        """Resolve a single input name from its MatchNames candidates, falling
        back to vernacular search when MatchNames found nothing."""

        source = ResolutionSource.MATCH_NAMES

        # No scientific match -> try common-name fallback.
        if not candidates:
            candidates = self._vernacular_lookup(input_name)
            source = ResolutionSource.VERNACULAR
            if not candidates:
                return None

        if len(candidates) == 1:
            return ResolvedTaxon.from_match_record(
                input_name, candidates[0], resolved_via=source, ambiguous=False
            )

        # Multiple candidates: rank them and decide whether the winner is clear.
        ranked = sorted(candidates, key=self._candidate_score, reverse=True)
        best, runner_up = ranked[0], ranked[1]

        gap = self._candidate_score(best) - self._candidate_score(runner_up)
        is_ambiguous = gap < _AMBIGUITY_GAP

        return ResolvedTaxon.from_match_record(
            input_name, best, resolved_via=source, ambiguous=is_ambiguous
        )

    # ------------------------------------------------------------------ #
    # Heuristic ranking
    # ------------------------------------------------------------------ #

    @staticmethod
    def _candidate_score(record: dict) -> int:
        """Heuristic score for picking among multiple candidates. Higher is better.

        Priority, highest weight first:
          1. accepted status        (a synonym is rarely what the user wants)
          2. exact match_type       (over phonetic / near matches)
          3. marine                 (this is a marine registry)
        The weights are spaced so a higher-priority signal always outranks all
        lower ones combined, making the ordering deterministic and explainable.
        """
        score = 0

        if TaxonStatus.from_worms(record.get("status")) == TaxonStatus.ACCEPTED:
            score += 100

        if (record.get("match_type") or "").lower() == "exact":
            score += 10

        if record.get("isMarine"):
            score += 1

        return score