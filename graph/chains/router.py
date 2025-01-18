from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field
from typing import Literal
load_dotenv()
class RouterQuerry(BaseModel):
    """
    Route a user querry to the most relevant datasource
    """
    datasource : Literal["vectorstore","websearch"]=Field(
        ...,
        description="Given a user question choose to route it to web search or vectorstore"
    )
llm = ChatOpenAI(temperature=0)

structured_llm_router =llm.wrap_model(RouterQuerry)
system_prompt = """
    you are an expert at routing user question to a vectorstore or websearch.
    The vectorestore contains documents related to agents, prompt engineering and adverserial attacks.
    User the vectorestore for questions on these topics. for all else use websearch.
"""
router_prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{question}")
]
)
question_router = router_prompt|structured_llm_router

if __name__ == '__main__':
    print(question_router.invoke(
        {"question":"what is the current weather in istanbul"}
    )
    )
