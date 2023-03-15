from langchain.llms import OpenAI


print ("*** NOTE: models, n, best_of ?? are standard across providers")
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
print(llm("Tell me a joke"))

print("*** NOTE: This returns Generation(text, genration_info{'finish_reason', 'logprobs'})")

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)
print (llm_result.generations[0])

print("*** NOTE: LLM alos outputs completion tokens, prompt tokens")
print(llm_result.llm_output)


