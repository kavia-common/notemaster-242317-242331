import os
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Internal helper to read an environment variable with an optional default."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_db_conninfo() -> str:
    """
    Build a Postgres connection string from environment variables.

    Expected env vars (provided by notes_database container):
    - POSTGRES_URL
    - POSTGRES_USER
    - POSTGRES_PASSWORD
    - POSTGRES_DB
    - POSTGRES_PORT

    POSTGRES_URL is expected to be a host-like value (NOT a full URL), but we also
    support full DSN strings if provided.
    """
    pg_url = _get_env("POSTGRES_URL")
    pg_user = _get_env("POSTGRES_USER")
    pg_password = _get_env("POSTGRES_PASSWORD")
    pg_db = _get_env("POSTGRES_DB")
    pg_port = _get_env("POSTGRES_PORT")

    # If user provides a full DSN, just use it.
    if pg_url and pg_url.startswith("postgres"):
        return pg_url

    # Otherwise, build from parts.
    if not (pg_url and pg_user and pg_password and pg_db and pg_port):
        missing = [
            k
            for k in ["POSTGRES_URL", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_PORT"]
            if not _get_env(k)
        ]
        raise RuntimeError(
            "Database is not configured. Missing environment variables: "
            + ", ".join(missing)
            + ". These should be provided by the notes_database container."
        )

    return f"postgresql://{pg_user}:{pg_password}@{pg_url}:{pg_port}/{pg_db}"


# PUBLIC_INTERFACE
def create_app() -> FastAPI:
    """
    Create and configure the NoteMaster FastAPI app.

    Returns:
        FastAPI: configured application instance with Notes API routes.
    """
    openapi_tags = [
        {"name": "Health", "description": "Service health checks."},
        {"name": "Notes", "description": "Create, edit, delete, list and search notes."},
    ]

    app = FastAPI(
        title="NoteMaster API",
        description=(
            "A small notes API supporting CRUD, tags, and full-text search.\n\n"
            "Database: PostgreSQL (notes table with tags[] and tsvector search)."
        ),
        version="1.0.0",
        openapi_tags=openapi_tags,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_conn() -> psycopg.Connection:
        """Create a new psycopg connection."""
        conninfo = _get_db_conninfo()
        return psycopg.connect(conninfo)

    def _ensure_schema() -> None:
        """
        Ensure DB schema exists.

        NOTE: This is a lightweight safety net. The database container is expected
        to initialize the schema as well.
        """
        schema_sqls = [
            """
            CREATE TABLE IF NOT EXISTS notes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                tags TEXT[] NOT NULL DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                search_tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
                ) STORED
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes (updated_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_notes_tags_gin ON notes USING GIN (tags);",
            "CREATE INDEX IF NOT EXISTS idx_notes_search_tsv ON notes USING GIN (search_tsv);",
        ]
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    for stmt in schema_sqls:
                        cur.execute(stmt)
                conn.commit()
        except Exception:
            # If env isn't set up yet, don't crash the app on startup;
            # endpoints will return a clear error when used.
            return

    @app.on_event("startup")
    async def _startup() -> None:
        _ensure_schema()

    # -----------------------
    # Models
    # -----------------------
    class NoteBase(BaseModel):
        title: str = Field(..., description="Note title.")
        content: str = Field("", description="Note content/body.")
        tags: List[str] = Field(default_factory=list, description="Optional tags for the note.")

    class NoteCreate(NoteBase):
        pass

    class NoteUpdate(BaseModel):
        title: Optional[str] = Field(None, description="New title.")
        content: Optional[str] = Field(None, description="New content/body.")
        tags: Optional[List[str]] = Field(None, description="Replace tags with this list.")

    class NoteOut(NoteBase):
        id: UUID = Field(..., description="Note ID (UUID).")
        created_at: datetime = Field(..., description="When the note was created.")
        updated_at: datetime = Field(..., description="When the note was last updated.")

    class NotesListOut(BaseModel):
        items: List[NoteOut] = Field(..., description="Notes list.")
        total: int = Field(..., description="Total notes matching the filters (before pagination).")

    def _db_available() -> None:
        """Raise 503 if DB env/config is missing."""
        try:
            _get_db_conninfo()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e

    # -----------------------
    # Routes
    # -----------------------
    @app.get(
        "/",
        tags=["Health"],
        summary="Health check",
        description="Basic health check endpoint.",
        operation_id="health_check",
    )
    def health_check() -> dict[str, str]:
        return {"message": "Healthy"}

    @app.get(
        "/notes",
        tags=["Notes"],
        summary="List notes",
        description="List notes, optionally filtered by text query and/or a tag, ordered by updated_at desc.",
        operation_id="list_notes",
        response_model=NotesListOut,
    )
    def list_notes(
        q: Optional[str] = Query(None, description="Search query (full text)."),
        tag: Optional[str] = Query(None, description="Filter notes containing this tag."),
        limit: int = Query(50, ge=1, le=200, description="Max notes to return."),
        offset: int = Query(0, ge=0, description="Pagination offset."),
        _: Any = Depends(_db_available),
    ) -> NotesListOut:
        where = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if q:
            where.append("search_tsv @@ plainto_tsquery('english', %(q)s)")
            params["q"] = q
        if tag:
            where.append("%(tag)s = ANY(tags)")
            params["tag"] = tag

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        list_sql = f"""
            SELECT id, title, content, tags, created_at, updated_at
            FROM notes
            {where_sql}
            ORDER BY updated_at DESC
            LIMIT %(limit)s OFFSET %(offset)s;
        """
        count_sql = f"SELECT COUNT(*) FROM notes {where_sql};"

        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(count_sql, params)
                total = int(cur.fetchone()[0])

                cur.execute(list_sql, params)
                rows = cur.fetchall()

        items = [
            NoteOut(
                id=row[0],
                title=row[1],
                content=row[2],
                tags=list(row[3] or []),
                created_at=row[4],
                updated_at=row[5],
            )
            for row in rows
        ]
        return NotesListOut(items=items, total=total)

    @app.get(
        "/notes/{note_id}",
        tags=["Notes"],
        summary="Get note",
        description="Get a note by ID.",
        operation_id="get_note",
        response_model=NoteOut,
    )
    def get_note(note_id: UUID, _: Any = Depends(_db_available)) -> NoteOut:
        sql = """
            SELECT id, title, content, tags, created_at, updated_at
            FROM notes
            WHERE id = %(id)s;
        """
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"id": note_id})
                row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found.")

        return NoteOut(
            id=row[0],
            title=row[1],
            content=row[2],
            tags=list(row[3] or []),
            created_at=row[4],
            updated_at=row[5],
        )

    @app.post(
        "/notes",
        tags=["Notes"],
        summary="Create note",
        description="Create a new note.",
        operation_id="create_note",
        response_model=NoteOut,
        status_code=status.HTTP_201_CREATED,
    )
    def create_note(payload: NoteCreate, _: Any = Depends(_db_available)) -> NoteOut:
        sql = """
            INSERT INTO notes (title, content, tags)
            VALUES (%(title)s, %(content)s, %(tags)s)
            RETURNING id, title, content, tags, created_at, updated_at;
        """
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    {"title": payload.title, "content": payload.content, "tags": payload.tags},
                )
                row = cur.fetchone()
            conn.commit()

        return NoteOut(
            id=row[0],
            title=row[1],
            content=row[2],
            tags=list(row[3] or []),
            created_at=row[4],
            updated_at=row[5],
        )

    @app.put(
        "/notes/{note_id}",
        tags=["Notes"],
        summary="Update note",
        description="Update an existing note. Only provided fields are changed.",
        operation_id="update_note",
        response_model=NoteOut,
    )
    def update_note(note_id: UUID, payload: NoteUpdate, _: Any = Depends(_db_available)) -> NoteOut:
        # Fetch current
        select_sql = """
            SELECT title, content, tags
            FROM notes
            WHERE id = %(id)s;
        """
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(select_sql, {"id": note_id})
                current = cur.fetchone()

                if not current:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found.")

                new_title = payload.title if payload.title is not None else current[0]
                new_content = payload.content if payload.content is not None else current[1]
                new_tags = payload.tags if payload.tags is not None else list(current[2] or [])

                update_sql = """
                    UPDATE notes
                    SET title = %(title)s,
                        content = %(content)s,
                        tags = %(tags)s,
                        updated_at = now()
                    WHERE id = %(id)s
                    RETURNING id, title, content, tags, created_at, updated_at;
                """
                cur.execute(
                    update_sql,
                    {
                        "id": note_id,
                        "title": new_title,
                        "content": new_content,
                        "tags": new_tags,
                    },
                )
                row = cur.fetchone()
            conn.commit()

        return NoteOut(
            id=row[0],
            title=row[1],
            content=row[2],
            tags=list(row[3] or []),
            created_at=row[4],
            updated_at=row[5],
        )

    @app.delete(
        "/notes/{note_id}",
        tags=["Notes"],
        summary="Delete note",
        description="Delete a note by ID.",
        operation_id="delete_note",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_note(note_id: UUID, _: Any = Depends(_db_available)) -> None:
        sql = "DELETE FROM notes WHERE id = %(id)s;"
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"id": note_id})
                deleted = cur.rowcount
            conn.commit()

        if deleted == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found.")

        return None

    @app.get(
        "/tags",
        tags=["Notes"],
        summary="List tags",
        description="Return all unique tags currently used by notes (sorted).",
        operation_id="list_tags",
        response_model=List[str],
    )
    def list_tags(_: Any = Depends(_db_available)) -> List[str]:
        sql = """
            SELECT DISTINCT unnest(tags) AS tag
            FROM notes
            WHERE array_length(tags, 1) IS NOT NULL
            ORDER BY tag ASC;
        """
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        return [r[0] for r in rows]

    return app


app = create_app()
