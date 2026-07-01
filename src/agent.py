from typing import Any, override, Optional, Literal
from pydantic import BaseModel, Field
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import asyncio
import os
from functools import lru_cache

from src.worms_api import WoRMS, MatchNamesParams
from src.logging import log_species_not_found
from src.tools import create_worms_tools
from src.util import get_llm_client_kwargs, update_llm_credentials

dotenv.load_dotenv()


class ToolPlan(BaseModel):
    tool_name: str
    priority: Literal["must_call", "should_call", "optional"]
    reason: str


class ResearchPlan(BaseModel):
    query_type: Literal["single_species", "comparison", "conservation", "distribution", "taxonomy"]
    species_mentioned: list[str]
    tools_planned: list[ToolPlan]
    reasoning: str


class MarineResearchParams(BaseModel):
    species_names: list[str] = Field(
        default=[],
        description="Scientific names of marine species to research",
        examples=[["Orcinus orca"], ["Orcinus orca", "Delphinus delphis"]]
    )


AGENT_DESCRIPTION = "Marine species research assistant using WoRMS database"


class WoRMSReActAgent(IChatBioAgent):
    def __init__(self):
        self.worms_logic = WoRMS()
        self._cached_lookup = lru_cache(maxsize=256)(
            self.worms_logic.get_species_aphia_id
        )
        
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
                    parameters=MarineResearchParams
                )
            ]
        )
    
    async def _create_plan(self, request: str, species_names: list[str]) -> ResearchPlan:
        parser = JsonOutputParser(pydantic_object=ResearchPlan)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a marine biology research planning expert.
Analyze queries and create structured execution plans.

Available tools:
- search_by_common_name: Convert common names to scientific (USE FIRST if common name)
- get_species_synonyms: Alternative scientific names for a species
- get_species_attributes: Conservation status, body size, IUCN, CITES, ecological traits
- get_attribute_definitions: Get the tree of attribute definitions (what types of data WoRMS can store)
- get_attribute_value_options: Get possible values for a specific attribute category
- get_taxonomic_record: Basic taxonomy (family, order, class)
- get_species_distribution: Geographic distribution/range
- get_vernacular_names: Common names in different languages
- get_taxonomic_classification: Full taxonomic hierarchy
- get_literature_sources: Scientific references and citations
- get_child_taxa: Child taxa/species under a taxonomic group
- get_external_ids: External database IDs (FishBase, NCBI, etc.)
- get_recent_species_changes: Species added/modified during a time period
- abort: Call if request cannot be fulfilled
- finish: Call when request is successfully completed

Query types:
- "single_species": Info about one species
- "comparison": Compare multiple species
- "conservation": Specifically about conservation/IUCN status
- "distribution": Specifically about where species lives
- "taxonomy": About classification

Tool priorities:
- "must_call": Required to answer the query
- "should_call": Recommended for complete answer
- "optional": Only if user specifically asks

{format_instructions}
"""),
            ("human", """Query: "{request}"
Species mentioned: {species}

Create the execution plan.""")
        ])
        
        llm_kwargs = get_llm_client_kwargs()
        llm = ChatOpenAI(
            model=os.getenv("LLM", "gpt-4.1-unfiltered"),
            temperature=0,
            openai_api_key=llm_kwargs["api_key"],
            openai_api_base=llm_kwargs["base_url"],
        )
        chain = prompt | llm | parser
        
        try:
            plan = await chain.ainvoke({
                "format_instructions": parser.get_format_instructions(),
                "request": request,
                "species": species_names if species_names else "unknown"
            })
            return ResearchPlan(**plan)
        except Exception as e:
            print(f"Warning: Plan creation failed ({e}), using fallback plan")
            
            tools_planned = [
                ToolPlan(
                    tool_name="get_species_attributes",
                    priority="must_call",
                    reason="Get ecological traits and conservation status"
                ),
                ToolPlan(
                    tool_name="get_taxonomic_record",
                    priority="should_call",
                    reason="Get basic taxonomy information"
                )
            ]
            
            return ResearchPlan(
                query_type="single_species" if len(species_names) <= 1 else "comparison",
                species_mentioned=species_names,
                tools_planned=tools_planned,
                reasoning="Fallback plan: get core species information"
            )

    async def _resolve_common_names_parallel(self, names: list[str], context: ResponseContext) -> dict[str, str]:
        async with context.begin_process(f"Resolving {len(names)} species names") as process:
            try:
                loop = asyncio.get_event_loop()
                
                match_params = MatchNamesParams(
                    scientific_names=names,
                    marine_only=True
                )
                api_url = self.worms_logic.build_match_names_url(match_params)
                
                await process.log(f"Batch matching {len(names)} names")
                
                raw_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    ),
                    timeout=30.0
                )
                
                if not isinstance(raw_response, list):
                    await process.log("Unexpected API response format")
                    return {}
                
                resolved = {}
                for input_name, matches in zip(names, raw_response):
                    if matches and len(matches) > 0:
                        best = matches[0]
                        scientific_name = best.get('scientificname')
                        match_type = best.get('match_type', 'unknown')
                        
                        resolved[input_name] = scientific_name
                        
                        if match_type == 'exact':
                            await process.log(f"'{input_name}' → {scientific_name} [exact match]")
                        else:
                            await process.log(f"'{input_name}' → {scientific_name} [fuzzy match: {match_type}]")
                    else:
                        await process.log(f"'{input_name}' → NOT FOUND")
                
                return resolved
                
            except asyncio.TimeoutError:
                await process.log(f"Batch resolution timed out after 30 seconds")
                return {}
            except Exception as e:
                await process.log(f"Batch resolution failed: {e}")
                return {}
    
    async def _get_cached_aphia_id(self, species_name: str, process) -> Optional[int]:
        loop = asyncio.get_event_loop()
        aphia_id = await loop.run_in_executor(
            None,
            self._cached_lookup,
            species_name
        )
        
        if aphia_id:
            await process.log(f"Resolved {species_name} -> AphiaID {aphia_id}")
        else:
            await log_species_not_found(process, species_name)
        
        return aphia_id
    
    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineResearchParams,
        metadata: dict[str, Any] | None = None,
    ):
        update_llm_credentials(metadata)
        async with context.begin_process("Searching WoRMS") as process:
            plan = await self._create_plan(request, params.species_names)
            
            species_str = ", ".join(plan.species_mentioned)
            await process.log(f"{plan.query_type.replace('_', ' ').title()} query: {species_str}")
            
            must_call_tools = [t.tool_name for t in plan.tools_planned if t.priority == "must_call"]
            should_call_tools = [t.tool_name for t in plan.tools_planned if t.priority == "should_call"]
            
            plan_details = f"Execution Plan: {len(must_call_tools)} required tools"
            if should_call_tools:
                plan_details += f", {len(should_call_tools)} recommended tools"
            
            await process.log(plan_details, data={
                "query_type": plan.query_type,
                "species_count": len(plan.species_mentioned),
                "must_call": must_call_tools,
                "should_call": should_call_tools,
                "reasoning": plan.reasoning
            })
            
            await context.reply(f"Researching {len(plan.species_mentioned)} species using {len(must_call_tools)} tools...")

        if plan.species_mentioned:
            async with context.begin_process("Resolving species names") as process:
                await process.log(f"Batch resolving {len(plan.species_mentioned)} name(s)")
                
                resolved = await self._resolve_common_names_parallel(plan.species_mentioned, context)
                
                await process.log(f"Resolved {len(resolved)}/{len(plan.species_mentioned)} species")
                
                for input_name, scientific_name in resolved.items():
                    aphia_id = await self._get_cached_aphia_id(scientific_name, process)
                    if not aphia_id:
                        await process.log(f"Warning: Could not cache AphiaID for {scientific_name}")

        tools = create_worms_tools(
            worms_logic=self.worms_logic,
            context=context,
            get_cached_aphia_id_func=self._get_cached_aphia_id
        )
        
        llm_kwargs = get_llm_client_kwargs()
        llm = ChatOpenAI(
            model=os.getenv("LLM", "gpt-4.1-unfiltered"),
            openai_api_key=llm_kwargs["api_key"],
            openai_api_base=llm_kwargs["base_url"],
        )
        system_prompt = self._make_system_prompt_with_plan(request, plan)
        agent = create_react_agent(llm, tools)
        
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=request)
                    ]
                }
            )
            # Agent execution completed
        except Exception as e:
            await context.reply(f"An error occurred: {str(e)}")
    
    def _make_system_prompt_with_plan(self, request: str, plan: ResearchPlan) -> str:
        must_call = [t for t in plan.tools_planned if t.priority == "must_call"]
        should_call = [t for t in plan.tools_planned if t.priority == "should_call"]
        
        tool_context = "\n\nEXECUTION PLAN:\n"
        tool_context += "MUST CALL (required to answer query):\n"
        for tool in must_call:
            tool_context += f"  • {tool.tool_name} - {tool.reason}\n"
        
        if should_call:
            tool_context += "\nSHOULD CALL (for complete answer):\n"
            for tool in should_call:
                tool_context += f"  • {tool.tool_name} - {tool.reason}\n"
        
        species_list = ", ".join(plan.species_mentioned) if plan.species_mentioned else "unknown"
        
        return f"""You are a marine biology research assistant with access to the WoRMS database.

USER REQUEST: "{request}"

QUERY TYPE: {plan.query_type}
STRATEGY: {plan.reasoning}
SPECIES: {species_list}
{tool_context}

CRITICAL INSTRUCTIONS:

1. FOLLOW THE EXECUTION PLAN ABOVE:
   - Call all "MUST CALL" tools (required for this query)
   - Call "SHOULD CALL" tools if they help provide a complete answer
   - Skip tools not listed in the plan
   - Call each tool AT MOST ONCE per species

2. NAME HANDLING:
   - If user provides common names, names have been pre-resolved via batch API
   - Use the scientific names for all tool calls
   - If resolution fails, tools will handle the lookup

3. FOR COMPARISON QUERIES:
   - Collect the SAME data points for all species
   - After collecting, provide comparative analysis with specific facts

4. TERMINATION (CRITICAL):
   - IMMEDIATELY call finish() after collecting all required data
   - DO NOT wait or hesitate - finish() as soon as you have the data
   - If tools return cached results, you ALREADY have the data
   - The finish() tool terminates execution - use it RIGHT AWAY

5. RESPONSE QUALITY:
   - Lead with direct answer to user's question
   - Include specific facts (IUCN status, sizes, locations)
   - Mention that full data is in artifacts

Always create artifacts when retrieving data from WoRMS.
"""


if __name__ == "__main__":
    agent = WoRMSReActAgent()
    print("=" * 60)
    print("WoRMS Agent Server")
    print("=" * 60)
    print(f"URL: http://localhost:9999")
    print(f"Status: Ready with planning capabilities")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)