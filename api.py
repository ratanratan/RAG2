
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import ask_question

app = FastAPI(
    title="Company HR RAG API",
    description="Ask questions about company policies.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=1000,
        description="Question about company policies"
    )


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        result = ask_question(request.question.strip())
        return result

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        ) from error

    except Exception as error:
        # Log the detailed error on the server,
        # but avoid exposing internal details to users.
        print(f"RAG request failed: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question. Please try again."
        ) from error
