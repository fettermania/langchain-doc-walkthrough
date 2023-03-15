from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI


llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb.total_tokens)


print("*** NOTE Contextm manager in get_openai_callback will track all tokens in scope")
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    result2 = llm("Tell me a joke")
    print(cb.total_tokens)




print("*** NOTE Contextm manager in get_openai_callback will track in chains too")

llm = OpenAI(temperature=0)

print("*** NOTE: serpapi requires signup at https://serpapi.com/, SERPAPI_API_KEY env, and 'pip install google-search-results'")
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


with get_openai_callback() as cb:
    response = agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
    print(cb.total_tokens)

    