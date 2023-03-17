# https://www.mlq.ai/getting-started-with-langchain/

import langchain
import openai
import os

from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
text = "What is a good name for an AI that helps you fine tune LLMs."
print(llm(text))

# PromptTemplate as just a formatter.  Ideally, we include examples.
"""
Language models take text as input - that text is commonly referred to as a prompt. Typically this is not simply a hardcoded string but rather a combination of a template, some examples, and user input. LangChain provides several classes and functions to make constructing and working with prompts easy.
"""

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

print(prompt.format(product="AI robots"))

# Run as a chain
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("AI robots"))

"""
Agents are systems that use a language model to interact with other tools. These can be used to do more grounded question/answering, interact with APIs, or even take actions.
"""


### Agent example
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Now let's test it out!
## NOTE: This doesn't find 3/3/2023 - it seems to find Today's on Google search
agent.run("What is Tesla's market capitalization as of March 3rd 2023, how many times bigger is this than Ford's market cap on the same day?")

## Memory
from langchain import ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="Hi there!")
conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
conversation.predict(input="My Name is The Grunt")
conversation.predict(input="Do you remember my name?")
conversation.predict(input="Do you still remember my name?")
