from langchain.utilities import SerpAPIWrapper
search = SerpAPIWrapper()
print(search.run("Obama's first name?"))


params = {
    "engine": "bing",
    "gl": "us",
    "hl": "en",
}
search = SerpAPIWrapper(params=params)
print(search.run("Obama's first name?"))
