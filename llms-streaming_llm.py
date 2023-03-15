from langchain.llms import OpenAI
#from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.schema import HumanMessage


llm = OpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
resp = llm("Write me a song about FourLoco in the style of Garrison Keillor")
print (resp)