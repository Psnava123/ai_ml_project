from langsmith import evaluate
from langchain_core.messages import HumanMessage
# Import the exact same graph object your Streamlit app already uses:
from agent import graph  # (Or wherever your compiled `graph` is created)

evaluate(
    lambda x: graph.invoke({"messages": [HumanMessage(content=x["question"])]}),
    data="langsmith_rag_dataset", # Type the exact name you gave your CSV dataset
)