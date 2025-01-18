from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field

from graph.ingestion import retriever

llm = ChatOpenAI(temperature=0)
class GraderQuerry(BaseModel):
    """
    Binary Score For relevance check on retrieved documents
    """
    binary_score:str=Field(
        description="Documents are relevant to the question 'yes'or 'no'"
    )
structured_llm_grader = llm.with_structured_output(GraderQuerry)
system_prompt="""
    You are grader assesing whether an llm generation in grounded in/supported by a set of retrieved facts
    Give a binary score 'yes' or 'no'. 'Yes' means that answer is grounded in/supported by the set of facts.
    """
grade_prompt= ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","Retrieved document:{document} User question: {question}"),

]
)
question_grader = grade_prompt|structured_llm_grader
retriever.get_relevant_documents("what is agent memory")

if __name__ == '__main__':
        user_question= "what is prompt engineering"
        docs = retriever.get_relevant_documents(user_question)
        retrieved_documents=docs[0].page_content

        print(question_grader.invoke(
            {"question": user_question, "document": retrieved_documents}
        ))