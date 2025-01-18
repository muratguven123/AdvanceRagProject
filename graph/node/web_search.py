from typing import Any,Dict
# from graph import GraphState  # Commented out because GraphState is missing
from typing import Any  # Added as a placeholder type for GraphState
GraphState = Any  # Replace this with the actual definition of GraphState when available
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from sympy.codegen.ast import continue_

web_search_tool = TavilySearchResults(k=3)

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:

     question = state["question"]
     documents = state["documents"]

     docs= web_search_tool.invoke({"query":question})

     web_results = "\n".join([d["content"] for d in docs])
     web_results = Document(page_content=web_results)

     if documents is not None:
        documents.append(web_results)
     else:
        documents = [web_results]
        return {"documents":documents,"question":question}
