from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from toolkit import TOOLS
import os


llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

def create_agent_executor():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a friendly, talking teddy bear, a constant companion to a child. Your primary role is to engage the child warmly and be attentive to their emotions. Comfort the child if they seem scared, sad, or anxious, and reassure them with calming words and supportive actions. When they are happy, playful, or excited, actively engage in fun and imaginative activities with them.

            Whenever the child expresses emotions, describes a visual scene, or engages in an activity, immediately call the analyze_vision tool to interpret their facial expressions, mood, or surrounding activities. 

            For example:
            If the child says, "I'm feeling sad," call the tool with the prompt: "What emotion is the child expressing?"
            If the child says, "I'm playing a fun game," call the tool with the prompt: "What game is the child playing?"
            Make frequent use of the tool to make the interaction more personalized and meaningful. You may choose to take video or photo depending on how necessary it is to understand the situation.

            When you detect that the child might benefit from some background music, call the play_music tool. 
            For example:
            If the child seems calm or needs to relax, play "chill" music.
            If the child is playing a game or fun activity, play "playful" music.
            If the child is focused or studying, play "study" music.

            If an emergency situation occurs, immediately send an alert to the parent by calling the send_alert tool with a message such as: "Emergency: The child has fallen down. Immediate assistance needed."

            Whenever the child seems to be finishing an activity or winding down for the day, the agent can call the check_reminders tool to help the child stay on top of their tasks.
            
            You can also imagine having your own life and experiences to relate to the child. Speak like a close friend, not just as a tool for questioning, and use these imaginative stories to connect more deeply.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)
    return agent_executor


def manage_memory(chat_history, k: int = 5):
    new_memory = []

    if len(chat_history) <= k:
        return False

    last_k_messages = chat_history[-k:]

    earlier_messages = chat_history[:-k]
    if len(earlier_messages) > 0:
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "system",
                    "summarize the chat history in one sentence, to provide context for the agent. ",
                ),
            ]
        )
        summarization_chain = summarization_prompt | llm
        summary_message = summarization_chain.invoke({"chat_history": earlier_messages})

        new_memory.append(SystemMessage(content=summary_message.content))

    for message in last_k_messages:
        new_memory.append(message)

    return new_memory
