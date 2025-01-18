from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field


class GradeAnswer(BaseModel):

    binary_score:str=Field(
        description="Documents are relevant to the question 'yes'or 'no'"
    )




llm = ChatOpenAI(temperature=0)

structured_llm_grader = llm.with_structured_output(GradeAnswer)
system_prompt="""
     You are a grader assesing whether an llm generation in grounded in/supported by a set of retrieved facts
     \n Give a binary score 'yes' or 'no'. 'Yes' means that answer is grounded in/supported by the set of facts.
     
"""
answer_prompt= ChatPromptTemplate.from_messages([

    ("system",system_prompt),
    ("human","{answer}")
])
answer_grader = answer_prompt|structured_llm_grader