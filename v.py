import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser



class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key="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",
            model_name="llama-3.3-70b-versatile"  
        )

    def recommend_movies(self, emotion):
        prompt = PromptTemplate.from_template(
            "Suggest 3 popular movies that match the mood '{emotion}'. Give a short 1-line explanation for each."
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"emotion": emotion})