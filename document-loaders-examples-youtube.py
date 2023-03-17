from langchain.document_loaders import YoutubeLoader
loader = YoutubeLoader.from_youtube_url("ttps://www.youtube.com/watch?v=FusOBwB61ds", add_video_info=True)
print(loader.load())

