import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.tools import Tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="LangGraph ChatBot", layout="centered")
st.title("📚🧠 LangGraph AI Chatbot")

# Check essential API keys
# groq_api_key = os.getenv("GROQ_API_KEY")
# serp_api_key = os.getenv("SERP_API_KEY")

groq_api_key = st.secrets['GROQ_API_KEY']
serp_api_key = st.secrets['SERP_API_KEY']

if not groq_api_key or not serp_api_key:
    st.error("Missing GROQ_API_KEY or SERP_API_KEY. Please set them in your environment.")
    st.stop()

# Initialize tools
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300))
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300))
serp_tool = Tool(
    name="SerpAPI Search",
    func=SerpAPIWrapper().run,
    description="Search the web using SerpAPI"
)
tools = [arxiv_tool, wiki_tool, serp_tool]

# LLM with tool binding
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
llm_with_tools = llm.bind_tools(tools=tools)

# LangGraph setup
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# Patch tool_call_id for ToolMessages
def patch_tool_messages(messages):
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and "tool_call_id" not in msg.additional_kwargs:
            msg.additional_kwargs["tool_call_id"] = f"tool_call_{i}"
    return messages

# Convert tuple history to LangChain message objects
def convert_history_to_lc_messages(history):
    messages = []
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "tool":
            messages.append(ToolMessage(content=content, tool_call_id="patched"))  # fallback
    return patch_tool_messages(messages)

# Initialize session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input UI
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Append user message
    st.session_state.chat_history.append(("user", user_input))

    # Convert history to LangChain messages
    lc_messages = convert_history_to_lc_messages(st.session_state.chat_history)

    # Run LangGraph
    events = graph.stream({"messages": lc_messages}, stream_mode="values")

    for event in events:
        message = event["messages"][-1]
        role = getattr(message, "type", "assistant")
        st.session_state.chat_history.append((role, message.content))
        with st.chat_message(role):
            st.markdown(message.content)

# Display full history
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)
