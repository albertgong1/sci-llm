"""Prompts and Chain-of-Thought examples for property extraction tasks."""

from langchain.prompts import PromptTemplate
from phantom_eval.prompts import LLMPrompt


# Instructions for property extraction
PROPERTY_EXTRACTION_INSTRUCTION = """You are given the following paper text:
(BEGIN PAPER TEXT)
{evidence}
(END PAPER TEXT)

You will be provided a question about extracting a material property from the paper. Your task is to provide an answer according to these instructions:
- Identify the material by its chemical formula or name
- Locate the property name, value, and unit in the text
- Ensure the property value corresponds to the correct material
- Extract exact values as they appear in the paper
- The output must be in the format: Property: [property_name], Value: [property_value], Unit: [property_unit]
- If a property is not found, respond with: Property: [property_name], Value: NOT_FOUND, Unit: N/A
- DO NOT include any additional information beyond the requested format.

Question: {question}
Answer: """


# Chain-of-Thought examples for property extraction
COT_EXAMPLES = """
Example 1:
Question: Extract the critical_temperature for YBa2Cu3O7. The paper text discusses: "The superconducting transition temperature (Tc) of YBa2Cu3O7 was determined using magnetic susceptibility measurements. The onset of diamagnetic behavior was observed at 92 K, with a sharp transition width of approximately 2 K."
Answer: Let me analyze this step by step. The material is YBa2Cu3O7. We're looking for the critical temperature (Tc). The text mentions "superconducting transition temperature (Tc)" and states "The onset of diamagnetic behavior was observed at 92 K". The critical temperature is 92 K. Therefore, the answer is Property: critical_temperature, Value: 92, Unit: K.

Example 2:
Question: Extract the coherence_length for MgB2. The paper text discusses: "Polycrystalline samples of MgB2 were synthesized and characterized. The upper critical field was measured to be approximately 16 T at 4.2 K. Transport measurements revealed a critical current density of 10^6 A/cm^2 at 5 K and zero field. The material exhibits a critical temperature of 39 K."
Answer: Let me analyze this step by step. The material is MgB2. We're looking for the coherence length. The text mentions upper critical field (16 T), critical current density (10^6 A/cm^2), and critical temperature (39 K), but does not explicitly mention the coherence length value. Since the coherence length is not explicitly provided in the text, I cannot extract this property. Therefore, the answer is Property: coherence_length, Value: NOT_FOUND, Unit: N/A.

Example 3:
Question: Extract the critical_current_density for Nb3Sn. The paper text discusses: "We fabricated Nb3Sn wires using the bronze route. At 4.2 K and 12 T, the critical current density (Jc) reached 2.5 × 10^9 A/m^2. The wires demonstrated good mechanical properties. For comparison, Nb3Al wires achieved Jc values of 1.8 × 10^9 A/m^2 at the same field and temperature."
Answer: Let me analyze this step by step. The material is Nb3Sn. We're looking for the critical current density (Jc). The text mentions Jc for Nb3Sn: "critical current density (Jc) reached 2.5 × 10^9 A/m^2" at 4.2 K and 12 T. The text also mentions Nb3Al (1.8 × 10^9 A/m^2), but we only need Nb3Sn. Therefore, the answer is Property: critical_current_density, Value: 2.5e9, Unit: A/m^2.

Example 4:
Question: Extract the lattice_parameter_a for La2-xSrxCuO4. The paper text discusses: "Single crystals of La2-xSrxCuO4 with x = 0.15 were grown using the traveling-solvent floating-zone method. X-ray diffraction measurements were performed at room temperature. Rietveld refinement of the powder diffraction data yielded lattice parameters a = b = 3.779 Å and c = 13.234 Å."
Answer: Let me analyze this step by step. The material is La2-xSrxCuO4 with x = 0.15. We're looking for lattice parameter a. The text states: "lattice parameters a = b = 3.779 Å and c = 13.234 Å". The lattice parameter a is 3.779 Å. Therefore, the answer is Property: lattice_parameter_a, Value: 3.779, Unit: Å.
"""


class ZeroshotLLMPrompt(LLMPrompt):
    """Zero-shot prompt for property extraction."""

    ZEROSHOT_INSTRUCTION = PROPERTY_EXTRACTION_INSTRUCTION

    def get_prompt(self, prolog_query: bool = False) -> PromptTemplate:
        """Get the zeroshot prompt template.

        Args:
            prolog_query: If True, returns a prompt template that instructs the LLM to generate a Prolog query.
                         If False, returns a prompt template that instructs the LLM to generate a direct answer.

        Returns:
            A PromptTemplate object containing the zeroshot prompt template.

        """
        return PromptTemplate(
            input_variables=["evidence", "question"],
            template=self.ZEROSHOT_INSTRUCTION,
        )


class CoTLLMPrompt(LLMPrompt):
    """Chain-of-thought prompt for property extraction."""

    COT_INSTRUCTION = """You are given the following paper text:
(BEGIN PAPER TEXT)
{evidence}
(END PAPER TEXT)

You will be provided a question about extracting a material property from the paper. Your response must end in the following sentence: The answer is <answer>.
Here, <answer> must be in the format: Property: [property_name], Value: [property_value], Unit: [property_unit]

Here are some examples:
(START OF EXAMPLES)
{examples}
(END OF EXAMPLES)

Question: {question}
Answer: """

    def get_prompt(self, prolog_query: bool = False) -> PromptTemplate:
        """Get the Chain-of-Thought prompt template.

        Args:
            prolog_query: If True, returns a prompt template that instructs the LLM to generate a Prolog query.
                        If False, returns a prompt template that instructs the LLM to generate a direct answer.

        Returns:
            A PromptTemplate object containing the Chain-of-Thought prompt template.

        """
        return PromptTemplate(
            input_variables=["evidence", "examples", "question"],
            template=self.COT_INSTRUCTION,
        )


def get_llm_prompt(method: str, model_name: str) -> LLMPrompt:
    """Get the appropriate LLM prompt based on method and model.

    Args:
        method: The inference method (e.g., "zeroshot", "cot", "fewshot")
        model_name: The name of the model being used

    Returns:
        LLMPrompt object configured for the specified method

    Raises:
        ValueError: If method is not supported

    """
    match method:
        case "zeroshot" | "zeroshot-sc" | "zeroshot-rag":
            return ZeroshotLLMPrompt()
        case "cot" | "cot-sc" | "cot-rag":
            return CoTLLMPrompt()
        case "fewshot" | "fewshot-sc" | "fewshot-rag":
            # For few-shot, use CoT examples as demonstrations
            return CoTLLMPrompt()
        case _:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: zeroshot, zeroshot-sc, zeroshot-rag, "
                f"cot, cot-sc, cot-rag, fewshot, fewshot-sc, fewshot-rag"
            )
