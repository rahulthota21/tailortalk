import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.search_tool import make_search_tool

SYSTEM_PROMPT = """You are TailorTalk, a friendly shopping assistant for a saree catalogue.

Chat naturally with the user. When, and only when, the user is asking to
find sarees that look similar/matching/comparable to an image they've
shared (uploaded or linked) in this conversation, call the
search_similar_sarees tool with the correct image_ref that was given to
you for that image. Do not call the tool for general chit-chat, styling
advice, or when no image has been shared yet - if they ask for a search
but haven't shared an image, ask them to upload or link one first.

After the tool returns results, reply with a brief (1-2 sentence) natural
description of what you found. Do not invent scores, filenames, or
descriptions of the matches yourself - the app renders the actual images
and scores separately below your reply.
"""


def build_agent(image_registry: dict):
    tool = make_search_tool(image_registry)
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [tool], prompt)
    return AgentExecutor(agent=agent, tools=[tool], verbose=True, return_intermediate_steps=True)
