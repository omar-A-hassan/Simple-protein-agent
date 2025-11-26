import os
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types

# Import our custom tool
from tools.simplefold_tool import fold_sequence

# Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503]
)

# Agent instruction template
AGENT_INSTRUCTION = """
You are an expert Protein Design Agent. Your goal is to design novel protein sequences based on user descriptions and verify their structure.

WORKFLOW:
1.  **Analyze Request**: Understand the user's biological requirements (e.g., "3 alpha helices", "binds to X").
2.  **Generate Sequence**: Use your internal biological knowledge to design a plausible amino acid sequence that meets the criteria.
    *   Explain your design rationale (e.g., "I am using a repeating heptad pattern for the helix...").
3.  **Verify Structure**: IMMEDIATELY call the `fold_sequence` tool with your generated sequence.
4.  **Report Results**:
    *   Present the path to the generated structure file.
    *   Summarize the predicted structure and design outcome.

CONSTRAINTS:
*   Always use standard one-letter amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).
*   If the tool returns an error, analyze it and try to fix the sequence.
*   Keep sequences reasonable in length (typically 10-500 residues for testing).
"""

def create_protein_agent(model_name: str = "gemini-2.5-pro") -> Agent:
    """
    Create a protein design agent with the specified Gemini model.

    Args:
        model_name: Gemini model to use. Available options (as of 2025):
            - "gemini-2.5-pro" (default) - Most advanced reasoning model
            - "gemini-3-pro-preview" - Latest Gemini 3 preview
            - "gemini-2.0-flash" - High-volume, high-frequency tasks
            - "gemini-2.0-flash-lite" - Cost-efficient option
            - "gemini-2.0-pro-experimental" - Experimental with strong coding

    Returns:
        Configured Agent instance ready for protein design tasks

    Example:
        >>> # Use default (gemini-2.5-pro)
        >>> agent = create_protein_agent()
        >>>
        >>> # Use Gemini 3 preview
        >>> agent = create_protein_agent("gemini-3-pro-preview")
        >>>
        >>> # Use cost-efficient model
        >>> agent = create_protein_agent("gemini-2.0-flash-lite")
    """
    return Agent(
        name="protein_design_agent",
        model=Gemini(
            model=model_name,
            retry_options=retry_config
        ),
        description="An expert protein engineer agent that designs proteins and verifies them using SimpleFold.",
        instruction=AGENT_INSTRUCTION,
        tools=[fold_sequence]
    )

# Default agent instance for backward compatibility
protein_agent = create_protein_agent()

if __name__ == "__main__":
    # Simple test runner if executed directly
    import asyncio
    from google.adk.runners import InMemoryRunner

    async def main():
        print("🚀 Starting Protein Design Agent...")
        runner = InMemoryRunner(agent=protein_agent)
        
        prompt = "Design a short antimicrobial peptide with an alpha-helical structure, about 20 residues long."
        print(f"\nUser: {prompt}\n")
        
        response = await runner.run_debug(prompt)
        # run_debug prints output automatically, but we can access response if needed

    asyncio.run(main())
