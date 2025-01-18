from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field

llm = ChatOpenAI(temperature=0)

class GradeHalucunations(BaseModel):
    "Binary Score For Halucunations present in generated answer"
    binary_score:str=Field(
        description="Documents are relevant to the question 'yes'or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHalucunations)
system_prompt="""
You are grader assesing whether an llm generation in grounded in/supported by a set of retrieved facts
    Give a binary score 'yes' or 'no'. 'Yes' means that answer is grounded in/supported by the set of facts.
    
"""
halucination_prompt= ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{answer},\n\n {documents}\n\n LLM generation:{generation}"),
])
halucination_grader = halucination_prompt|structured_llm_grader