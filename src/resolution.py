"""
Resolution contract for the WoRMS agent.

Everything in WoRMS hangs off the AphiaID, so preprocessing's job is to turn
whatever the user supplied (a scientific name, a misspelling, a common name)
into a validated AphiaID *before* any data tool runs. These models are the
contract between the resolver (preprocessing) and the tools that consume it.

Field names mirror the WoRMS REST `AphiaRecordsByMatchNames` response
(AphiaID, status, valid_AphiaID, match_type, etc.), normalized to snake_case.

A note on the two IDs every resolved taxon carries:
  - `matched_aphia_id`  : the AphiaID of the name WoRMS matched.
  - `accepted_aphia_id` : the AphiaID of the *accepted* taxon to actually query.
When a user names an unaccepted synonym (e.g. "Orca" -> AphiaID 380520,
status "unaccepted"), WoRMS hands back the accepted target inline as
`valid_AphiaID` (137021, "Orcinus"). Data lookups must use the accepted ID,
or they query a dead synonym that carries little or no data.
`effective_aphia_id` is the single value tools should use.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def portal_url(aphia_id: int) -> str:
    """Human-facing WoRMS landing page for an AphiaID."""
    return f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"


class TaxonStatus(str, Enum):
    """Taxonomic status as reported by WoRMS in the `status` field."""
    ACCEPTED = "accepted"
    UNACCEPTED = "unaccepted"
    # WoRMS also emits values like "nomen dubium", "temporary name", etc.
    # OTHER captures anything outside the two we branch on.
    OTHER = "other"

    @classmethod
    def from_worms(cls, raw: Optional[str]) -> "TaxonStatus":
        if not raw:
            return cls.OTHER
        normalized = raw.strip().lower()
        if normalized == "accepted":
            return cls.ACCEPTED
        if normalized == "unaccepted":
            return cls.UNACCEPTED
        return cls.OTHER


class ResolutionSource(str, Enum):
    """Which WoRMS endpoint resolved this taxon (provenance for the handoff)."""
    MATCH_NAMES = "match_names"          # AphiaRecordsByMatchNames (primary)
    VERNACULAR = "vernacular"            # AphiaRecordsByVernacular (common-name fallback)


class ResolvedTaxon(BaseModel):
    """One successfully resolved taxon: the bridge from a user-supplied name
    to the AphiaID the tools will query."""

    # --- what the user/agent gave us ---
    input_name: str = Field(
        ...,
        description="The name as supplied by the user or agent (may be a misspelling or common name).",
        examples=["Orca", "killer whale", "Carcharodon carcharias"],
    )

    # --- what WoRMS matched ---
    matched_name: str = Field(
        ...,
        description="The scientificname WoRMS matched for the input.",
        examples=["Orca", "Carcharodon carcharias"],
    )
    matched_aphia_id: int = Field(
        ...,
        description="AphiaID of the matched name (may be an unaccepted synonym).",
        examples=[380520, 105838],
    )
    status: TaxonStatus = Field(
        ...,
        description="Taxonomic status of the matched name.",
    )
    unaccept_reason: Optional[str] = Field(
        None,
        description="WoRMS `unacceptreason` when status is unaccepted (e.g. 'preoccupied').",
        examples=["preoccupied"],
    )

    # --- the accepted taxon to actually query ---
    accepted_aphia_id: int = Field(
        ...,
        description="AphiaID of the accepted taxon (WoRMS `valid_AphiaID`).",
        examples=[137021, 105838],
    )
    accepted_name: str = Field(
        ...,
        description="Name of the accepted taxon (WoRMS `valid_name`).",
        examples=["Orcinus", "Carcharodon carcharias"],
    )

    # --- descriptive metadata (straight from the match record) ---
    rank: Optional[str] = Field(None, description="Taxonomic rank, e.g. 'Species', 'Genus'.")
    authority: Optional[str] = Field(None, description="Authorship string, e.g. '(Linnaeus, 1758)'.")
    match_type: Optional[str] = Field(
        None,
        description="WoRMS match quality: 'exact', 'phonetic', 'near_1', etc.",
        examples=["exact"],
    )
    is_marine: Optional[bool] = Field(None, description="WoRMS `isMarine` flag (1 -> True).")

    # --- resolution provenance / flags ---
    resolved_via: ResolutionSource = Field(
        ...,
        description="Which endpoint produced this resolution.",
    )
    redirected: bool = Field(
        False,
        description="True when the matched name differs from the accepted taxon "
                    "(i.e. the user named a synonym and we pivoted).",
    )
    ambiguous: bool = Field(
        False,
        description="True when WoRMS returned multiple candidates and one was selected.",
    )

    @property
    def effective_aphia_id(self) -> int:
        """The AphiaID tools should use for all data lookups: the accepted taxon."""
        return self.accepted_aphia_id

    @property
    def portal_url(self) -> str:
        """Landing page for the accepted taxon (for artifact metadata / handoff)."""
        return portal_url(self.accepted_aphia_id)

    def handoff_note(self) -> Optional[str]:
        """A short, non-influencing note for the assistant when something about the
        resolution is worth surfacing (synonym pivot, fuzzy match). Returns None when
        the resolution was clean and exact, so we don't add noise."""
        if self.redirected:
            return (
                f"'{self.input_name}' matched the unaccepted name '{self.matched_name}'"
                f"{f' ({self.unaccept_reason})' if self.unaccept_reason else ''}; "
                f"used accepted taxon '{self.accepted_name}' (AphiaID {self.accepted_aphia_id})."
            )
        if self.match_type and self.match_type != "exact":
            return (
                f"'{self.input_name}' resolved by {self.match_type} match to "
                f"'{self.accepted_name}' (AphiaID {self.accepted_aphia_id})."
            )
        return None

    @classmethod
    def from_match_record(
        cls,
        input_name: str,
        record: dict,
        resolved_via: ResolutionSource = ResolutionSource.MATCH_NAMES,
        ambiguous: bool = False,
    ) -> "ResolvedTaxon":
        """Build a ResolvedTaxon from a single WoRMS match/vernacular record.

        Handles the accepted-name pivot: when `status` is unaccepted, the
        accepted taxon is taken from `valid_AphiaID` / `valid_name`.
        """
        status = TaxonStatus.from_worms(record.get("status"))

        matched_aphia_id = record.get("AphiaID")
        valid_aphia_id = record.get("valid_AphiaID") or matched_aphia_id
        valid_name = record.get("valid_name") or record.get("scientificname")

        is_marine_raw = record.get("isMarine")
        is_marine = bool(is_marine_raw) if is_marine_raw is not None else None

        return cls(
            input_name=input_name,
            matched_name=record.get("scientificname", input_name),
            matched_aphia_id=matched_aphia_id,
            status=status,
            unaccept_reason=record.get("unacceptreason"),
            accepted_aphia_id=valid_aphia_id,
            accepted_name=valid_name,
            rank=record.get("rank"),
            authority=record.get("authority"),
            match_type=record.get("match_type"),
            is_marine=is_marine,
            resolved_via=resolved_via,
            redirected=(matched_aphia_id != valid_aphia_id),
            ambiguous=ambiguous,
        )


class ResolutionResult(BaseModel):
    """Outcome of resolving a batch of input names: the resolved taxa plus the
    names that could not be resolved (which drive a clean abort/clarification)."""

    resolved: list[ResolvedTaxon] = Field(
        default_factory=list,
        description="Successfully resolved taxa.",
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="Input names that matched nothing (after vernacular fallback).",
    )

    @property
    def all_resolved(self) -> bool:
        return not self.unresolved

    @property
    def any_resolved(self) -> bool:
        return bool(self.resolved)

    def aphia_id_map(self) -> dict[str, int]:
        """The contract tools consume: input_name -> effective (accepted) AphiaID.

        Keyed by the original input_name so a tool invoked with whatever the user
        said can look up the right AphiaID without re-resolving.
        """
        return {t.input_name: t.effective_aphia_id for t in self.resolved}

    def handoff_notes(self) -> list[str]:
        """Non-empty resolution notes (synonym pivots, fuzzy matches) for the assistant."""
        return [note for t in self.resolved if (note := t.handoff_note())]