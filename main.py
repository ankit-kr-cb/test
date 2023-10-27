from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from cb import Chatbot

app = FastAPI()


class QueryInput(BaseModel):
    query: str


def get_chatbot():
    return Chatbot()


@app.post("/chatbot_response/")
def chatbot_response(query_input: QueryInput, chatbot: Chatbot = Depends(get_chatbot)):
    response = chatbot.convchain(query_input.query)
    return {"response": response}


@app.post("/clear_history/")
def clear_history(chatbot: Chatbot = Depends(get_chatbot)):
    chatbot.clr_history()
    return {"message": "Chat history cleared"}
