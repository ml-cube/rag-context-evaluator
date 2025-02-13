from pydantic import BaseModel, Field


class RagRatingResponse(BaseModel):
    rating: int = Field(description="The relevance rating for the context")
    explanation: str = Field(description="The explanation for the rating")
