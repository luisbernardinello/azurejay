from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    first_name: str
    last_name: str
    

class UserProfile(BaseModel):
    user_id: UUID
    user_name: str | None = None
    user_difficulties: list[str] = []
    user_interests: list[str] = []
    
class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    new_password_confirm: str
