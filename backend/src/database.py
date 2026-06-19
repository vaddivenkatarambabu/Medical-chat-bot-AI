import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DEFAULT_SQLITE_PATH = Path("instance") / "medicore.sqlite3"

load_dotenv()


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()

    if not url:
        DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

    if url.startswith("postgres://"):
        url = f"postgresql://{url.removeprefix('postgres://')}"

    if url.startswith("postgresql://"):
        url = f"postgresql+psycopg://{url.removeprefix('postgresql://')}"

    return url


def is_sqlite_url(url: str | None = None) -> bool:
    return (url or get_database_url()).startswith("sqlite")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _engine_kwargs(url: str) -> dict:
    kwargs: dict = {
        "future": True,
        "pool_pre_ping": True,
    }

    if is_sqlite_url(url):
        kwargs["connect_args"] = {"check_same_thread": False}
        return kwargs

    kwargs.update(
        {
            "pool_size": _env_int("DATABASE_POOL_SIZE", 5, minimum=1),
            "max_overflow": _env_int("DATABASE_MAX_OVERFLOW", 10, minimum=0),
            "pool_timeout": _env_int("DATABASE_POOL_TIMEOUT", 30, minimum=1),
            "pool_recycle": _env_int("DATABASE_POOL_RECYCLE", 1800, minimum=1),
        }
    )
    return kwargs


DATABASE_URL = get_database_url()
engine: Engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def should_auto_create_schema() -> bool:
    configured = os.getenv("DATABASE_AUTO_CREATE")
    if configured is not None:
        return configured == "1"

    return is_sqlite_url()


def init_database() -> None:
    if not should_auto_create_schema():
        return

    from src import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def check_database() -> bool:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True
