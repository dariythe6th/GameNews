import os
import time
import threading
import numpy as np
from typing import List, Optional, Generator
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ИСПРАВЛЕНИЕ: Импорт асинхронных компонентов напрямую из SQLAlchemy
# (SQLModel использует эти же компоненты)
from sqlmodel import Field, SQLModel, select, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select as sa_select

# Elasticsearch
from elasticsearch import Elasticsearch, helpers, exceptions as es_exceptions

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------
# CONFIG
# ---------------------------------------

# Асинхронный URL для FastAPI (требует 'asyncpg')
POSTGRES_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/gamenews"
)

# Синхронный URL для ML (требует 'psycopg2' или 'psycopg')
SYNC_POSTGRES_URL = POSTGRES_URL.replace("+asyncpg", "")

ES_URL = os.environ.get("ELASTIC_URL", "http://localhost:9200")
ES_INDEX = os.environ.get("ES_INDEX", "news_index")
APP_HOST = "0.0.0.0"
APP_PORT = 8080


# ---------------------------------------
# DATABASE MODELS (ОБНОВЛЕНО: __table_args__)
# ---------------------------------------

class User(SQLModel, table=True):
    # ДОБАВЛЕНО: Это позволяет Uvicorn Reloader переимпортировать модель без ошибки.
    __table_args__ = {'extend_existing': True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    preferred_genres: Optional[str] = ""


class News(SQLModel, table=True):
    # ДОБАВЛЕНО:
    __table_args__ = {'extend_existing': True}

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    genres: Optional[str] = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    views: int = 0


class Interaction(SQLModel, table=True):
    # ДОБАВЛЕНО:
    __table_args__ = {'extend_existing': True}

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int
    news_id: int
    event: str
    ts: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------
# PYDANTIC OUTPUT/INPUT MODELS (Без изменений)
# ---------------------------------------

class NewsOut(BaseModel):
    id: int
    title: str
    content: str
    genres: Optional[str]
    created_at: datetime
    views: int
    score: float
    read: bool


class CreateNewsIn(BaseModel):
    title: str
    content: str
    genres: Optional[str] = ""


# ---------------------------------------
# APP INITIALIZATION & DEPENDENCIES
# ---------------------------------------

app = FastAPI(title="Game News API — FastAPI + ML + Elasticsearch (Async)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DATABASE
# Асинхронный движок для FastAPI (основной)
async_engine = create_async_engine(POSTGRES_URL, echo=False)

# Синхронный движок для ML/фоновых задач
sync_engine = create_engine(SYNC_POSTGRES_URL, echo=False)


# Асинхронная зависимость для сессии
async def get_session() -> Generator[AsyncSession, None, None]:
    async with AsyncSession(async_engine) as session:
        yield session


# Инициализация базы данных
async def init_db():
    async with async_engine.begin() as conn:
        # SQLModel.metadata находится в синхронном контексте, поэтому run_sync
        await conn.run_sync(SQLModel.metadata.create_all)


# Запускаем инициализацию при старте приложения
@app.on_event("startup")
async def on_startup():
    await init_db()
    ensure_es_index()


# ---------------------------------------
# ELASTICSEARCH INIT (Без изменений)
# ---------------------------------------

es = Elasticsearch(ES_URL, verify_certs=False)


def ensure_es_index():
    """Проверяет и создает индекс ES (синхронная операция)."""
    try:
        if not es.indices.exists(index=ES_INDEX):
            es.indices.create(
                index=ES_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "genres": {"type": "keyword"},
                            "views": {"type": "integer"},
                            "created_at": {"type": "date"}
                        }
                    }
                }
            )
    except Exception as e:
        print(f"Elasticsearch not ready or error during index check: {e}")


# ---------------------------------------
# ML – In-Memory TF-IDF Model (Без изменений)
# ---------------------------------------

ml_lock = threading.Lock()
tfidf_vectorizer: Optional[TfidfVectorizer] = None
news_ids: List[int] = []
news_tfidf_matrix: Optional[np.ndarray] = None


def rebuild_tfidf():
    """Перестраивает TF-IDF матрицу из всех новостей (синхронная операция)."""
    global tfidf_vectorizer, news_ids, news_tfidf_matrix

    with ml_lock:
        with sync_engine.begin() as session:
            rows = session.execute(sa_select(News)).scalars().all()

        if not rows:
            tfidf_vectorizer = None
            news_ids = []
            news_tfidf_matrix = None
            return

        docs = []
        ids = []

        for n in rows:
            text = " ".join([n.title] * 3 + [n.content, n.genres or ""])
            docs.append(text)
            ids.append(n.id)

        vectorizer = TfidfVectorizer(max_features=6000, stop_words="russian")
        X = vectorizer.fit_transform(docs)

        tfidf_vectorizer = vectorizer
        news_tfidf_matrix = X
        news_ids = ids
        print("TF-IDF matrix rebuilt successfully.")


# ---------------------------------------
# HELPERS (Без изменений)
# ---------------------------------------

async def get_user_read_set(user_id: int, session: AsyncSession) -> set:
    """Получает набор ID новостей, прочитанных пользователем (асинхронно)."""
    stmt = select(Interaction.news_id).where(
        Interaction.user_id == user_id,
        Interaction.event == "view"
    )
    result = await session.execute(stmt)
    return set(result.scalars().all())


def user_profile_vector(user_id: int, read_ids: set) -> Optional[np.ndarray]:
    """Строит профиль пользователя на основе прочитанных новостей (синхронно)."""
    global tfidf_vectorizer, news_tfidf_matrix, news_ids

    if tfidf_vectorizer is None or news_tfidf_matrix is None:
        return None

    with ml_lock:
        id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
        idxs = [id_to_idx[n] for n in read_ids if n in id_to_idx]

        if not idxs:
            return None

        mat = news_tfidf_matrix[idxs].toarray()
        vec = mat.mean(axis=0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec


async def compute_scores(candidates: List[int], profile_vec: Optional[np.ndarray], session: AsyncSession):
    """Гибридный скоринг (асинхронный для части БД-запроса)."""
    global news_ids, news_tfidf_matrix

    if news_tfidf_matrix is None or tfidf_vectorizer is None:
        return {nid: 0.0 for nid in candidates}

    with ml_lock:
        id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
        idxs = [id_to_idx[n] for n in candidates if n in id_to_idx]

        if not idxs:
            return {}

        mat = news_tfidf_matrix[idxs].toarray()

        if profile_vec is not None:
            sims = cosine_similarity(mat, profile_vec.reshape(1, -1)).reshape(-1)
            sims = (sims + 1) / 2
        else:
            sims = np.zeros(len(idxs))

    stmt = select(News.id, News.views).where(News.id.in_(candidates))
    result = await session.execute(stmt)
    views_map = {nid: view for nid, view in result}

    views = np.array([views_map.get(nid, 0) for nid in candidates], dtype=float)

    if views.max() > 0:
        log_views = np.log1p(views)
        views_norm = log_views / log_views.max()
    else:
        views_norm = np.zeros_like(views)

    scores = 0.75 * sims + 0.25 * views_norm

    return {candidates[i]: float(scores[i]) for i in range(len(candidates))}


def rebuild_all():
    """Полное обновление ES и TF-IDF (синхронная фоновая задача)."""
    with sync_engine.begin() as session:
        all_news = session.execute(sa_select(News)).scalars().all()

    actions = []
    for n in all_news:
        actions.append({
            "_op_type": "index",
            "_index": ES_INDEX,
            "_id": str(n.id),
            "_source": {
                "title": n.title,
                "content": n.content,
                "genres": (n.genres or "").split(","),
                "created_at": n.created_at.isoformat(),
                "views": n.views
            }
        })

    try:
        helpers.bulk(es, actions, refresh=True, chunk_size=1000)
    except es_exceptions.ConnectionError as e:
        print(f"ES connection error during bulk indexing: {e}")
    except Exception as e:
        print(f"ES indexing error: {e}")

    rebuild_tfidf()


# ---------------------------------------
# API ROUTES (Без изменений)
# ---------------------------------------

@app.post("/api/news/init-data")
async def init_data(background: BackgroundTasks, session: AsyncSession = Depends(get_session)):
    """Создаёт пользователей + тестовые новости."""
    user_count = (await session.execute(select(User))).all()
    if not user_count:
        session.add_all([
            User(name="Игрок 1 (Default)", preferred_genres="RPG,Strategy"),
            User(name="Игрок 2", preferred_genres="Action,Shooter"),
            User(name="Игрок 3", preferred_genres="Strategy,Simulation"),
        ])

    news_count = (await session.execute(select(News))).all()
    if not news_count:
        session.add_all([
            News(title="Сезон 5 стартует",
                 content="Новый сезон приносит много обновлений в геймплей и механики RPG.",
                 genres="RPG,Action"),
            News(title="Большие скидки",
                 content="Скидки на скины, наборы и подписку. Поспешите купить до конца недели.",
                 genres="Shop,Event"),
            News(title="Турнир выходного дня",
                 content="Участвуйте и выигрывайте награды в нашем еженедельном турнире по шутерам.",
                 genres="Shooter,Competitive"),
            News(title="Секреты стратегии",
                 content="Как побеждать в стратегии: гайд для новичков и опытных игроков.",
                 genres="Strategy,Guide"),
            News(title="Новое оружие",
                 content="Обзор нового мощного оружия для шутеров, которое изменит мету.",
                 genres="Shooter,Update"),
        ])

    await session.commit()
    background.add_task(rebuild_all)
    return {"status": "ok", "message": "Initialization started."}


@app.get("/api/news/feed", response_model=List[NewsOut])
async def feed(
        userId: int,
        limit: int = Query(20, gt=0, le=100),
        session: AsyncSession = Depends(get_session)
):
    """Формирует персонализированную ленту новостей."""
    user = await session.get(User, userId)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {userId} not found")

    read_ids = await get_user_read_set(userId, session)

    candidate_ids = []
    try:
        res = es.search(
            index=ES_INDEX,
            body={"query": {"match_all": {}}, "size": 200, "sort": [{"created_at": "desc"}]}
        )
        hits = res["hits"]["hits"]
        candidate_ids = [int(h["_id"]) for h in hits]
    except:
        print("Warning: ES search failed. Falling back to DB.")
        candidate_ids = (await session.execute(select(News.id))).scalars().all()

    unseen_candidate_ids = [nid for nid in candidate_ids if nid not in read_ids]

    if not unseen_candidate_ids:
        return []

    profile_vec = user_profile_vector(userId, read_ids)
    scores = await compute_scores(unseen_candidate_ids, profile_vec, session)

    sorted_ids = sorted(
        unseen_candidate_ids,
        key=lambda nid: scores.get(nid, 0.0),
        reverse=True
    )[:limit]

    news_map = {}
    if sorted_ids:
        stmt = select(News).where(News.id.in_(sorted_ids))
        news_list = (await session.execute(stmt)).scalars().all()
        news_map = {n.id: n for n in news_list}

    out = []
    for nid in sorted_ids:
        n = news_map.get(nid)
        if n:
            out.append(
                NewsOut(
                    id=n.id,
                    title=n.title,
                    content=n.content,
                    genres=n.genres,
                    created_at=n.created_at,
                    views=n.views,
                    score=float(scores.get(nid, 0.0)),
                    read=False
                )
            )

    return out


@app.post("/api/news/{news_id}/view")
async def view(news_id: int, userId: int, session: AsyncSession = Depends(get_session)):
    """Регистрирует просмотр новости и увеличивает счетчик views."""
    user = await session.get(User, userId)
    news = await session.get(News, news_id)

    if not user or not news:
        raise HTTPException(status_code=404, detail="User or News not found")

    session.add(Interaction(user_id=userId, news_id=news_id, event="view"))
    news.views += 1
    session.add(news)
    await session.commit()
    await session.refresh(news)

    try:
        es.update(
            index=ES_INDEX,
            id=str(news_id),
            body={"doc": {"views": news.views}},
            refresh=True
        )
    except Exception as e:
        print(f"ES update error: {e}")

    return {"status": "ok", "views": news.views}


@app.get("/api/news/health")
def health():
    """Проверка работоспособности."""
    return {"status": "ok"}


# ---------------------------------------
# RUN
# ---------------------------------------

if __name__ == "__main__":
    import uvicorn

    # ИСПРАВЛЕНИЕ: Передаем строку импорта "main:app" вместо переменной 'app'
    uvicorn.run(
        "main:app",  # <-- Строка импорта
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
        log_level="info"  # Рекомендуется для лучшего логирования
    )