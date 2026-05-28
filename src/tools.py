import asyncio
from typing import Callable
from functools import wraps
from langchain.tools import tool
from src.worms_api import  (
    SynonymsParams,
    DistributionParams,
    VernacularParams,
    SourcesParams,
    RecordParams,
    ClassificationParams,
    ChildrenParams,
    ExternalIDParams,
    AttributesParams,
    VernacularSearchParams,
    AttributeKeysParams,
    AttributeValuesByCategoryParams,
    RecordsByDateParams
)
from src.Wormslogging import (
    log_species_not_found,
    log_api_call,
    log_data_fetched,
    log_no_data,
    log_tool_error,
    log_artifact_created
)


def _portal_url(aphia_id: int) -> str:
    """Human-facing WoRMS landing page for an AphiaID (goes in artifact metadata)."""
    return f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"


def create_worms_tools(worms_logic, context, get_cached_aphia_id_func: Callable):
    # Per-run cache of SUCCESSFUL tool results. Keyed by tool name + all args so
    # different species/inputs never collide. Only successful retrievals are stored
    # (see cache_tool_result), so a transient failure is never cached as "no data".
    tool_call_tracker = {}

    def create_tracked_key(tool_name: str, *args, **kwargs) -> str:
        """Unique key for tool + arguments. Includes positional AND keyword args so
        e.g. get_species_synonyms('Orcinus orca') and ('Delphinus delphis') differ."""
        parts = [tool_name]
        parts.extend(str(a) for a in args)
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(parts)

    def cache_tool_result(func):
        """Cache successful results and RETURN them so the ReAct agent can reason over
        the status. Failures (errors / no-data / species-not-found) are not cached, so
        a later retry can still succeed. Success is identified by the 'Retrieved' prefix
        that every tool uses for its success status string."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            call_key = create_tracked_key(func.__name__, *args, **kwargs)

            if call_key in tool_call_tracker:
                # Already retrieved this exact call earlier in the run — hand the agent
                # the same status so it knows the artifact already exists.
                return tool_call_tracker[call_key]

            result = await func(*args, **kwargs)

            # Only cache confirmed successes. Failure/no-data strings are left uncached.
            if isinstance(result, str) and result.startswith("Retrieved"):
                tool_call_tracker[call_key] = result

            return result

        return wrapper

    async def get_species_or_fail(species_name: str, process) -> tuple[int | None, str | None]:
        """
        Get AphiaID for species or return error message.
        Returns: (aphia_id, error_message) - one will always be None
        """
        aphia_id = await get_cached_aphia_id_func(species_name, process)
        if not aphia_id:
            await log_species_not_found(process, species_name)
            return None, f"Species '{species_name}' not found in WoRMS database."
        return aphia_id, None

    async def fetch_paginated_data(process, api_url_func, batch_size=50):
        """
        Fetch all data with pagination support.
        api_url_func should be a function that takes offset and returns the API URL
        """
        loop = asyncio.get_event_loop()
        all_data = []
        offset = 1

        while True:
            api_url = api_url_func(offset)

            raw_response = await loop.run_in_executor(
                None,
                lambda url=api_url: worms_logic.execute_request(url)
            )

            batch = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

            if not batch:
                break

            all_data.extend(batch)

            if len(batch) < batch_size:
                break

            offset += batch_size

        return all_data

    # ------------------------------------------------------------------ #
    # Control tools (terminate the ReAct loop). reply() addresses the
    # iChatBio assistant, not the user.
    # ------------------------------------------------------------------ #

    @tool(return_direct=True)
    async def abort(reason: str):
        """Call if you cannot fulfill the request. Provide a clear reason why."""
        await context.reply(f"Unable to complete request: {reason}")

    @tool(return_direct=True)
    async def finish(summary: str):
        """Call when the request is successfully completed. Summarize WHAT was retrieved
        and that the data is available in the created artifacts (do not invent facts)."""
        await context.reply(summary)

    # ------------------------------------------------------------------ #
    # Data tools. Each returns a short retrieval-STATUS string for the
    # ReAct loop; the actual data lives in the artifact (uris=[api_url]),
    # with the human-facing portal URL in metadata.
    # ------------------------------------------------------------------ #

    @tool
    @cache_tool_result
    async def get_species_synonyms(species_name: str) -> str:
        """Get synonyms and alternative scientific names for a marine species."""
        async with context.begin_process(f"Searching WoRMS for synonyms of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                # Paginated fetch with offset
                def url_builder(offset):
                    syn_params = SynonymsParams(aphia_id=aphia_id)
                    api_url = worms_logic.build_synonyms_url(syn_params)
                    if offset > 1:
                        separator = '&' if '?' in api_url else '?'
                        api_url = f"{api_url}{separator}offset={offset}"
                    return api_url

                base_url = url_builder(1)
                await log_api_call(process, "get_species_synonyms", species_name, aphia_id, base_url)

                all_synonyms = await fetch_paginated_data(process, url_builder)

                if not all_synonyms:
                    await log_no_data(process, "get_species_synonyms", species_name, aphia_id)
                    return f"No synonyms found for {species_name}"

                await log_data_fetched(process, "get_species_synonyms", species_name, len(all_synonyms))

                syn_params = SynonymsParams(aphia_id=aphia_id)
                base_api_url = worms_logic.build_synonyms_url(syn_params)
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Synonyms for {species_name} (AphiaID: {aphia_id}) - {len(all_synonyms)} records",
                    uris=[base_api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "count": len(all_synonyms),
                        "species": species_name,
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_species_synonyms", species_name)
                return f"Retrieved {len(all_synonyms)} synonym record(s) for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_species_synonyms", species_name, e)
                return f"Error retrieving synonyms: {str(e)}"

    @tool
    @cache_tool_result
    async def get_species_distribution(species_name: str) -> str:
        """Get geographic distribution and range data for a marine species. Shows where the species is found globally."""
        async with context.begin_process(f"Searching WoRMS for distribution of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                dist_params = DistributionParams(aphia_id=aphia_id)
                api_url = worms_logic.build_distribution_url(dist_params)

                await log_api_call(process, "get_species_distribution", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                distributions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not distributions:
                    await log_no_data(process, "get_species_distribution", species_name, aphia_id)
                    return f"No distribution data found for {species_name}"

                await log_data_fetched(process, "get_species_distribution", species_name, len(distributions))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Distribution for {species_name} (AphiaID: {aphia_id}) - {len(distributions)} locations",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "count": len(distributions),
                        "species": species_name,
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_species_distribution", species_name)
                return f"Retrieved {len(distributions)} distribution record(s) for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_species_distribution", species_name, e)
                return f"Error retrieving distribution: {str(e)}"

    @tool
    @cache_tool_result
    async def get_vernacular_names(species_name: str) -> str:
        """Get common names for a marine species in different languages. Useful for finding local or colloquial names."""
        async with context.begin_process(f"Searching WoRMS for vernacular names of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                vern_params = VernacularParams(aphia_id=aphia_id)
                api_url = worms_logic.build_vernacular_url(vern_params)

                await log_api_call(process, "get_vernacular_names", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                vernaculars = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not vernaculars:
                    await log_no_data(process, "get_vernacular_names", species_name, aphia_id)
                    return f"No vernacular names found for {species_name}"

                await log_data_fetched(process, "get_vernacular_names", species_name, len(vernaculars))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Vernacular names for {species_name} (AphiaID: {aphia_id}) - {len(vernaculars)} names",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "count": len(vernaculars),
                        "species": species_name,
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_vernacular_names", species_name)
                return f"Retrieved {len(vernaculars)} vernacular name(s) for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_vernacular_names", species_name, e)
                return f"Error retrieving vernacular names: {str(e)}"

    @tool
    @cache_tool_result
    async def get_literature_sources(species_name: str) -> str:
        """Get scientific literature sources, references, and citations for a marine species. Provides academic sources."""
        async with context.begin_process(f"Searching WoRMS for literature sources of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                sources_params = SourcesParams(aphia_id=aphia_id)
                api_url = worms_logic.build_sources_url(sources_params)

                await log_api_call(process, "get_literature_sources", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                sources = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not sources:
                    await log_no_data(process, "get_literature_sources", species_name, aphia_id)
                    return f"No literature sources found for {species_name}"

                await log_data_fetched(process, "get_literature_sources", species_name, len(sources))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Literature sources for {species_name} (AphiaID: {aphia_id}) - {len(sources)} sources",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "count": len(sources),
                        "species": species_name,
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_literature_sources", species_name)
                return f"Retrieved {len(sources)} literature source(s) for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_literature_sources", species_name, e)
                return f"Error retrieving literature sources: {str(e)}"

    @tool
    @cache_tool_result
    async def get_taxonomic_record(species_name: str) -> str:
        """Get basic taxonomic record including family, order, class, status, and authority. Good for quick taxonomy overview."""
        async with context.begin_process(f"Searching WoRMS for taxonomic record of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                record_params = RecordParams(aphia_id=aphia_id)
                api_url = worms_logic.build_record_url(record_params)

                await log_api_call(process, "get_taxonomic_record", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                if not raw_response:
                    await log_no_data(process, "get_taxonomic_record", species_name, aphia_id)
                    return f"No taxonomic record found for {species_name}"

                await log_data_fetched(process, "get_taxonomic_record", species_name, 1)

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Taxonomic record for {species_name} (AphiaID: {aphia_id})",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "species": species_name,
                        "rank": raw_response.get('rank', '') if isinstance(raw_response, dict) else '',
                        "status": raw_response.get('status', '') if isinstance(raw_response, dict) else '',
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_taxonomic_record", species_name)
                return f"Retrieved taxonomic record for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_taxonomic_record", species_name, e)
                return f"Error retrieving taxonomic record: {str(e)}"

    @tool
    @cache_tool_result
    async def get_taxonomic_classification(species_name: str) -> str:
        """Get full taxonomic classification hierarchy from kingdom to species. Shows complete taxonomic tree."""
        async with context.begin_process(f"Searching WoRMS for classification of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                class_params = ClassificationParams(aphia_id=aphia_id)
                api_url = worms_logic.build_classification_url(class_params)

                await log_api_call(process, "get_taxonomic_classification", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                if not raw_response:
                    await log_no_data(process, "get_taxonomic_classification", species_name, aphia_id)
                    return f"No classification found for {species_name}"

                await log_data_fetched(process, "get_taxonomic_classification", species_name, 1)

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Taxonomic classification for {species_name} (AphiaID: {aphia_id})",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "species": species_name,
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_taxonomic_classification", species_name)
                return f"Retrieved taxonomic classification for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_taxonomic_classification", species_name, e)
                return f"Error retrieving classification: {str(e)}"

    @tool
    @cache_tool_result
    async def get_child_taxa(species_name: str) -> str:
        """Get child taxa (subspecies, varieties) under a taxonomic group. Useful for finding related species."""
        async with context.begin_process(f"Searching WoRMS for child taxa of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                children_params = ChildrenParams(aphia_id=aphia_id)
                api_url = worms_logic.build_children_url(children_params)

                await log_api_call(process, "get_child_taxa", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                children = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not children:
                    await log_no_data(process, "get_child_taxa", species_name, aphia_id)
                    return f"No child taxa found for {species_name}"

                await log_data_fetched(process, "get_child_taxa", species_name, len(children))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Child taxa for {species_name} (AphiaID: {aphia_id}) - {len(children)} children",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "count": len(children),
                        "species": species_name,
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_child_taxa", species_name)
                return f"Retrieved {len(children)} child taxa for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_child_taxa", species_name, e)
                return f"Error retrieving child taxa: {str(e)}"

    @tool
    @cache_tool_result
    async def get_external_ids(species_name: str, id_type: str = None) -> str:
        """Get external database identifiers (FishBase, NCBI, ITIS, BOLD, GISD). Links species to other databases."""
        async with context.begin_process(f"Searching WoRMS for external IDs of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                ext_params = ExternalIDParams(aphia_id=aphia_id, id_type=id_type)
                api_url = worms_logic.build_external_id_url(ext_params)

                await log_api_call(process, "get_external_ids", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                external_ids = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not external_ids:
                    await log_no_data(process, "get_external_ids", species_name, aphia_id)
                    id_type_str = f" of type '{id_type}'" if id_type else ""
                    return f"No external IDs{id_type_str} found for {species_name}"

                await log_data_fetched(process, "get_external_ids", species_name, len(external_ids))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"External database IDs for {species_name} (AphiaID: {aphia_id}) - {len(external_ids)} IDs",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "count": len(external_ids),
                        "species": species_name,
                        "id_type": id_type or "all",
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_external_ids", species_name)
                id_type_str = f" of type '{id_type}'" if id_type else ""
                return f"Retrieved {len(external_ids)} external ID(s){id_type_str} for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_external_ids", species_name, e)
                return f"Error retrieving external IDs: {str(e)}"

    @tool
    @cache_tool_result
    async def get_species_attributes(species_name: str) -> str:
        """Get ecological attributes, traits, and characteristics (IUCN status, body size, depth range, habitat). Provides conservation and ecological data."""
        async with context.begin_process(f"Searching WoRMS for attributes of {species_name}") as process:
            try:
                aphia_id, error = await get_species_or_fail(species_name, process)
                if error:
                    return error

                loop = asyncio.get_event_loop()
                attr_params = AttributesParams(aphia_id=aphia_id)
                api_url = worms_logic.build_attributes_url(attr_params)

                await log_api_call(process, "get_species_attributes", species_name, aphia_id, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                attributes = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not attributes:
                    await log_no_data(process, "get_species_attributes", species_name, aphia_id)
                    return f"No ecological attributes found for {species_name}"

                await log_data_fetched(process, "get_species_attributes", species_name, len(attributes))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Ecological attributes for {species_name} (AphiaID: {aphia_id}) - {len(attributes)} attributes",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "count": len(attributes),
                        "species": species_name,
                        "attribute_types": list(set([a.get('measurementType', '') for a in attributes if isinstance(a, dict)])),
                        "landing_page": _portal_url(aphia_id),
                    }
                )

                await log_artifact_created(process, "get_species_attributes", species_name)
                return f"Retrieved {len(attributes)} ecological attribute(s) for {species_name}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_species_attributes", species_name, e)
                return f"Error retrieving attributes: {str(e)}"

    @tool
    @cache_tool_result
    async def search_by_common_name(common_name: str) -> str:
        """Search for species using common names like 'killer whale' or 'great white shark'. Returns matching species with scientific names."""
        async with context.begin_process(f"Searching WoRMS for species with common name '{common_name}'") as process:
            try:
                loop = asyncio.get_event_loop()
                search_params = VernacularSearchParams(vernacular_name=common_name, like=True)
                api_url = worms_logic.build_vernacular_search_url(search_params)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                results = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not results:
                    await process.log(f"No species found with common name '{common_name}'")
                    return f"No species found with common name '{common_name}'. Try a different name or use scientific name."

                await process.log(f"Found {len(results)} species matching '{common_name}'")

                top_result = results[0].get('scientificname', '') if results and isinstance(results[0], dict) else ''

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Search results for common name '{common_name}' - {len(results)} species found",
                    uris=[api_url],
                    metadata={
                        "search_term": common_name,
                        "count": len(results),
                        "top_result": top_result,
                    }
                )

                await log_artifact_created(process, "search_by_common_name", common_name)
                return f"Retrieved {len(results)} species matching '{common_name}'. Artifact created."

            except Exception as e:
                await process.log(f"Error searching for common name '{common_name}': {type(e).__name__} - {str(e)}")
                return f"Error searching for common name: {str(e)}"

    @tool
    @cache_tool_result
    async def get_attribute_definitions(attribute_id: int = 0, include_children: bool = True) -> str:
        """Get the tree of available attribute types in WoRMS. Shows what ecological data categories exist (use attribute_id=0 for root)."""
        async with context.begin_process(f"Searching WoRMS for attribute definitions (ID: {attribute_id})") as process:
            try:
                loop = asyncio.get_event_loop()
                keys_params = AttributeKeysParams(attribute_id=attribute_id, include_children=include_children)
                api_url = worms_logic.build_attribute_keys_url(keys_params)

                await log_api_call(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", None, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                definitions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not definitions:
                    await log_no_data(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", None)
                    return f"No attribute definitions found for ID {attribute_id}"

                await log_data_fetched(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", len(definitions))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Attribute definitions from WoRMS (ID: {attribute_id}) - {len(definitions)} definition(s)",
                    uris=[api_url],
                    metadata={
                        "attribute_id": attribute_id,
                        "include_children": include_children,
                        "count": len(definitions)
                    }
                )

                await log_artifact_created(process, "get_attribute_definitions", f"Attribute ID {attribute_id}")
                return f"Retrieved {len(definitions)} attribute definition(s) for ID {attribute_id}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", e)
                return f"Error retrieving attribute definitions: {str(e)}"

    @tool
    @cache_tool_result
    async def get_attribute_value_options(category_id: int) -> str:
        """Get possible values for a specific attribute category. Use after get_attribute_definitions to find valid options."""
        async with context.begin_process(f"Searching WoRMS for attribute values in category {category_id}") as process:
            try:
                loop = asyncio.get_event_loop()
                values_params = AttributeValuesByCategoryParams(category_id=category_id)
                api_url = worms_logic.build_attribute_values_by_category_url(values_params)

                await log_api_call(process, "get_attribute_value_options", f"Category {category_id}", None, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                values = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not values:
                    await log_no_data(process, "get_attribute_value_options", f"Category {category_id}", None)
                    return f"No attribute values found for category {category_id}"

                await log_data_fetched(process, "get_attribute_value_options", f"Category {category_id}", len(values))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Attribute values for category {category_id} - {len(values)} value(s)",
                    uris=[api_url],
                    metadata={
                        "category_id": category_id,
                        "count": len(values)
                    }
                )

                await log_artifact_created(process, "get_attribute_value_options", f"Category {category_id}")
                return f"Retrieved {len(values)} attribute value(s) for category {category_id}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_attribute_value_options", f"Category {category_id}", e)
                return f"Error retrieving attribute values: {str(e)}"

    @tool
    @cache_tool_result
    async def get_recent_species_changes(start_date: str, end_date: str = None, marine_only: bool = True, extant_only: bool = True, offset: int = 1, max_results: int = 50) -> str:
        """Get species added or modified in WoRMS during a date range. Useful for tracking new discoveries and taxonomic updates. Use ISO 8601 format (e.g., '2024-01-01T00:00:00+00:00')."""
        async with context.begin_process(f"Searching WoRMS for species changes since {start_date}") as process:
            try:
                loop = asyncio.get_event_loop()
                date_params = RecordsByDateParams(
                    startdate=start_date,
                    enddate=end_date,
                    marine_only=marine_only,
                    extant_only=extant_only,
                    offset=offset
                )
                api_url = worms_logic.build_records_by_date_url(date_params)

                await log_api_call(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", None, api_url)

                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )

                records = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not records:
                    await log_no_data(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", None)
                    return f"No species changes found in WoRMS since {start_date}"

                total_found = len(records)
                if len(records) > max_results:
                    await process.log(f"Limiting results to {max_results} out of {total_found} found")
                    records = records[:max_results]

                await log_data_fetched(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", len(records))

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Species changes since {start_date} - {len(records)} records" + (f" (limited from {total_found})" if total_found > max_results else ""),
                    uris=[api_url],
                    metadata={
                        "start_date": start_date,
                        "end_date": end_date or "today",
                        "marine_only": marine_only,
                        "extant_only": extant_only,
                        "offset": offset,
                        "count": len(records),
                        "total_found": total_found,
                        "limited_to": max_results
                    }
                )

                await log_artifact_created(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}")
                limited_note = f" (limited from {total_found})" if total_found > max_results else ""
                return f"Retrieved {len(records)} species change record(s) for the given date range{limited_note}. Artifact created."

            except Exception as e:
                await log_tool_error(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", e)
                return f"Error retrieving recent changes: {str(e)}"

    return [
        get_species_synonyms,
        get_species_distribution,
        get_vernacular_names,
        get_literature_sources,
        get_taxonomic_record,
        get_taxonomic_classification,
        get_child_taxa,
        get_external_ids,
        get_species_attributes,
        get_attribute_definitions,
        get_attribute_value_options,
        get_recent_species_changes,
        search_by_common_name,
        abort,
        finish
    ]