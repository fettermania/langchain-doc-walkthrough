from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
wolfram = WolframAlphaAPIWrapper()
print(wolfram.run("What is 2x+5 = -3x + 7?"))
