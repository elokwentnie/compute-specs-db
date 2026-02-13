"""
Database models and configuration for Compute Specs DB

Uses SQLAlchemy ORM with SQLite database for storing compute specification data.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./cpu_database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CPUSpec(Base):
    """Compute specification database model"""
    __tablename__ = "cpu_specs"

    id = Column(Integer, primary_key=True, index=True)
    cpu_model_name = Column(String, index=True)
    family = Column(String)
    cpu_model = Column(String)
    codename = Column(String, index=True)
    cores = Column(Integer)
    threads = Column(Integer)
    tdp_watts = Column(Integer)
    launch_year = Column(Integer)
    max_turbo_frequency_ghz = Column(Float)
    l3_cache_mb = Column(Float)
    max_memory_tb = Column(Float)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Database session dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
