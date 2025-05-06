from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    first_name: str
    last_name: str
    

class UserProfile(BaseModel):
    id: UUID
    first_name: str
    user_difficulties: list[str] = []
    user_interests: list[str] = []
    
class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    new_password_confirm: str
