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
    user_profile: str
    user_difficulties: str
    user_interests: str
    

class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    new_password_confirm: str
