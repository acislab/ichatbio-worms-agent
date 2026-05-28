from typing import override, Optional

from pydantic import BaseModel, Field

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langsmith import traceable

import dotenv
import asyncio

from src.worms_api import WoRMS
from src.resolver import WoRMSResolver
from src.resolution import ResolutionResult
from src.Wormslogging import log_species_not_found
from src.tools import create_worms_tools

dotenv.load_dotenv()


class MarineResearchParams(BaseModel):
    species_names: list[str] = Field(
        default=[],
        description="Scientific or common names of marine species to research",
        examples=[["Orcinus orca"], ["Orcinus orca", "Delphinus delphis"], ["killer whale"]],
    )


AGENT_DESCRIPTION = "Marine species research assistant using WoRMS database"


class WoRMSReActAgent(IChatBioAgent):
    def __init__(self):
        self.worms_logic = WoRMS()
        # Preprocessing: one resolver, sharing the same WoRMS HTTP client.
        self._resolver = WoRMSResolver(self.worms_logic)

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="WoRMS Agent",
            description=AGENT_DESCRIPTION,
            icon="https://www.marinespecies.org/images/WoRMS_logo.png",
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="research_marine_species",
                    description=AGENT_DESCRIPTION,
                    parameters=MarineResearchParams,
                )
            ],
        )

    # ------------------------------------------------------------------ #
    # Preprocessing: resolve input names to accepted AphiaIDs, once.
    # ------------------------------------------------------------------ #

    async def _resolve_names(self, names: list[str], context: ResponseContext) -> ResolutionResult:
        """Run the WoRMS resolver once, up front, and log the outcome. The resolver
        is blocking (HTTP), so it runs in an executor."""
        async with context.begin_process(f"Resolving {len(names)} species name(s)") as process:
            loop = asyncio.get_event_loop()
            result: ResolutionResult = await loop.run_in_executor(
                None, self._resolver.resolve, names
            )

            for taxon in result.resolved:
                await process.log(
                    f"'{taxon.input_name}' -> {taxon.accepted_name} (AphiaID {taxon.accepted_aphia_id})",
                    data={
                        "input_name": taxon.input_name,
                        "accepted_name": taxon.accepted_name,
                        "accepted_aphia_id": taxon.accepted_aphia_id,
                        "matched_aphia_id": taxon.matched_aphia_id,
                        "status": taxon.status.value,
                        "match_type": taxon.match_type,
                        "redirected": taxon.redirected,
                        "ambiguous": taxon.ambiguous,
                        "resolved_via": taxon.resolved_via.value,
                    },
                )

            for note in result.handoff_notes():
                await process.log(note)

            if result.unresolved:
                await process.log(
                    f"Could not resolve: {', '.join(result.unresolved)}",
                    data={"unresolved": result.unresolved},
                )

            return result

    def _make_aphia_lookup(self, resolution: Optional[ResolutionResult]):
        """Build the name->AphiaID lookup the tools use.

        Map-first (covers everything pre-resolved, including the synonym pivot),
        with a live fallback for names the ReAct agent invents mid-reasoning that
        weren't in the original request (e.g. a family name it decides to explore).
        Keyed case-insensitively by input/accepted/matched name so a hit doesn't
        depend on which form the agent passes."""
        resolved_map: dict[str, int] = {}
        if resolution:
            for taxon in resolution.resolved:
                for key in (taxon.input_name, taxon.accepted_name, taxon.matched_name):
                    if key:
                        resolved_map[key.strip().lower()] = taxon.effective_aphia_id

        async def get_aphia_id(species_name: str, process) -> Optional[int]:
            key = (species_name or "").strip().lower()

            # 1. Pre-resolved path (no network).
            if key in resolved_map:
                aphia_id = resolved_map[key]
                await process.log(f"Using pre-resolved AphiaID {aphia_id} for '{species_name}'")
                return aphia_id

            # 2. Live fallback for agent-invented names not in the original request.
            loop = asyncio.get_event_loop()
            aphia_id = await loop.run_in_executor(
                None, self.worms_logic.get_species_aphia_id, species_name
            )
            if aphia_id:
                await process.log(f"Resolved '{species_name}' -> AphiaID {aphia_id} (live lookup)")
            else:
                await log_species_not_found(process, species_name)
            return aphia_id

        return get_aphia_id

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    @override
    @traceable(name="worms_agent_run", run_type="chain")
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineResearchParams,
    ):
        # --- Preprocessing: resolve names once, up front ---
        resolution: Optional[ResolutionResult] = None
        if params.species_names:
            resolution = await self._resolve_names(params.species_names, context)

            # Clean abort: if the user named species and NONE resolved, stop here
            # rather than letting the agent march into tools that will 204.
            if not resolution.any_resolved:
                await context.reply(
                    f"I could not resolve any of the requested names in WoRMS: "
                    f"{', '.join(resolution.unresolved)}. Please check the spelling, "
                    f"or try a scientific name.",
                    data={"unresolved": resolution.unresolved},
                )
                return

        # --- Build tools backed by the resolved AphiaIDs ---
        get_aphia_id = self._make_aphia_lookup(resolution)
        tools = create_worms_tools(
            worms_logic=self.worms_logic,
            context=context,
            get_cached_aphia_id_func=get_aphia_id,
        )

        # --- ReAct agent (no separate planner; the loop plans and acts) ---
        llm = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(llm, tools, prompt=self._make_system_prompt(request, resolution))

        run_metadata = {
            "user_query": request,
            "species_count": len(params.species_names),
            "resolved_count": len(resolution.resolved) if resolution else 0,
        }

        try:
            await agent.ainvoke(
                {"messages": [HumanMessage(content=request)]},
                config={"metadata": run_metadata, "run_name": "worms_react"},
            )
        except Exception as e:
            await context.reply(f"An error occurred while researching: {str(e)}")

    # ------------------------------------------------------------------ #
    # System prompt (soft guidance, not an enforced plan)
    # ------------------------------------------------------------------ #

    def _make_system_prompt(self, request: str, resolution: Optional[ResolutionResult]) -> str:
        if resolution and resolution.resolved:
            lines = [
                f"- '{t.input_name}' -> {t.accepted_name} (AphiaID {t.accepted_aphia_id})"
                for t in resolution.resolved
            ]
            resolved_block = "RESOLVED SPECIES (use these accepted names in tool calls):\n" + "\n".join(lines)

            notes = resolution.handoff_notes()
            if notes:
                resolved_block += "\n\nRESOLUTION NOTES (mention these to the assistant when summarizing):\n"
                resolved_block += "\n".join(f"- {n}" for n in notes)

            if resolution.unresolved:
                resolved_block += (
                    f"\n\nUNRESOLVED (could not be found in WoRMS): "
                    f"{', '.join(resolution.unresolved)}"
                )
        else:
            resolved_block = "No species were pre-resolved; resolve names with the tools as needed."

        return f"""You are a marine biology research assistant that queries the WoRMS \
(World Register of Marine Species) database through tools.

USER REQUEST: "{request}"

{resolved_block}

HOW TO WORK:
- Choose tools based on what the request needs: attributes for conservation/IUCN/CITES/size, \
distribution for where a species lives, classification or taxonomic record for taxonomy, \
synonyms/vernaculars/external IDs/sources as relevant.
- Prefer the accepted scientific names listed above when calling tools.
- For comparisons, retrieve the SAME data types for each species so they can be compared.
- Each tool stores its results as an artifact; the actual data lives in those artifacts for \
downstream processing. Tool replies are short status strings, NOT the data itself.

WHEN DONE:
- Call finish() with a brief summary of WHAT you retrieved (which data types, for which \
species) and note that artifacts were created. Surface any resolution notes above. \
Do NOT fabricate specific facts — the data is in the artifacts, not in your summary.
- If the request cannot be fulfilled, call abort() with a clear reason.
"""


if __name__ == "__main__":
    agent = WoRMSReActAgent()
    print("=" * 60)
    print("WoRMS Agent Server")
    print("=" * 60)
    print("URL: http://localhost:9999")
    print("Status: Ready (resolver preprocessing + ReAct)")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)