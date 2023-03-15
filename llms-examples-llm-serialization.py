from langchain.llms import OpenAI
from langchain.llms.loading import load_llm

print("*** NOTE: LLMs can be saved in JSON or yaml.  Check out the default config of text-davinci-003")
llm = load_llm("llm.json")
print(llm)