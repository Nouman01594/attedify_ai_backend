from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres:admin123@localhost:5432/attendify_db"

engine = create_engine(
    DATABASE_URL,
    echo=True  # VERY IMPORTANT (shows SQL in terminal)
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()
