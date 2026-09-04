"""
Schémas de validation pour l'authentification.
"""

from pydantic import BaseModel, Field, EmailStr


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="Adresse email")
    password: str = Field(..., min_length=6)
    full_name: str | None = None
    role: str = Field(default="viewer", pattern="^(admin|analyst|viewer)$")


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str | None
    role: str
    is_active: bool

    model_config = {"from_attributes": True}
