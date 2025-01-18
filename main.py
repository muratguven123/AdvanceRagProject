from dotenv import load_dotenv

load_dotenv()

import graph.node.graph

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(graph.app.invoke(input={"question": "agent memory?"}))