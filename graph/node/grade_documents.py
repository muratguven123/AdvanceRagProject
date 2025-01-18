from typing import Any,Dict
from graph.chains.retrieval import retrival_grader
from graph.state import GraphState
from langchain.chains.hyde.prompts import web_search
from sympy.codegen.ast import continue_


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relivant to the question.
    if Any Document is not relevant, we wişş set a flag to run web search
    Arg:
    state(dict): The Current state of the graph

    Returns:
        state(dict):Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANT TO QUESTİON")
    question = state["question"]
    documents = state["documents"]

    web_search= False
    for d in documents:
        score = retrival_grader.invoke(
            {"question":question,"document":d.page_content}
        )
        grade = score.binary_score

    if grade.lower()=="yes":

       print("----GRADE:DOCUMENT")
    else:
        print("----GRADE:DOCUMENT NOT RELEVANT" )
        web_search=True
        continue_

        return {"question":question,"documents":documents,"web_search":web_search}
