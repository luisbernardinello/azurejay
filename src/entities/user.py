from sqlalchemy import Column, String, ARRAY
from sqlalchemy.dialects.postgresql import UUID
import uuid
from typing import List
from ..database.core import Base


class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    
    # user Profile

    user_difficulties = Column(ARRAY(String), nullable=True, default=[])
    user_interests = Column(ARRAY(String), nullable=True, default=[])
    
    def __repr__(self):
        return f"<User(email='{self.email}', first_name='{self.first_name}', last_name='{self.last_name}')>"