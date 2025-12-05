import os
import csv
from datetime import datetime
from sqlmodel import Field, SQLModel, create_engine, Session, select
from sqlalchemy.future import select as sa_select

# ---------------------------------------
# CONFIG
# ---------------------------------------

# Синхронный URL для загрузчика (используем SYNC_POSTGRES_URL из main.py)
POSTGRES_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/gamenews"
)
SYNC_POSTGRES_URL = POSTGRES_URL.replace("+asyncpg", "")

# ---------------------------------------
# DATABASE MODELS (Дублируем из main.py для корректной работы)
# ---------------------------------------

class User(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: int = Field(default=None, primary_key=True)
    name: str
    preferred_genres: str = ""

class News(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: int = Field(default=None, primary_key=True)
    title: str
    content: str
    genres: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    views: int = 0

# ---------------------------------------
# MAIN LOAD FUNCTION
# ---------------------------------------

def load_data():
    """Загружает данные из CSV и создает тестовых пользователей."""
    print("Starting data loading...")

    # Используем синхронный движок для Session
    engine = create_engine(SYNC_POSTGRES_URL, echo=False)

    try:
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        print(f"Error creating tables: {e}")
        return

    with Session(engine) as session:
        # 1. Загрузка пользователей
        user_count = session.exec(select(User)).all()
        if not user_count:
            print("Creating default users...")
            session.add_all([
                User(name="Игрок 1 (Default - RPG/Strategy)", preferred_genres="RPG,Strategy"),
                User(name="Игрок 2 (Action/Shooter)", preferred_genres="Action,Shooter"),
                User(name="Игрок 3 (Simulation)", preferred_genres="Simulation,Casual"),
            ])
            session.commit()
            print(f"Created {len(session.exec(select(User)).all())} users.")
        else:
             print(f"Users already exist ({len(user_count)}). Skipping user creation.")


        # 2. Загрузка новостей из CSV
        news_count = session.exec(select(News)).all()
        if not news_count:
            print("Loading news from data.csv...")
            news_items = []
            try:
                with open('data.csv', 'r', encoding='utf-8') as f:
                    # Используем DictReader для работы с заголовками
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Удаляем кавычки, если они остались после csv-формата
                        content = row['content'].strip('"')
                        news_items.append(News(
                            title=row['title'],
                            content=content,
                            genres=row['genres'].replace(' ', ''), # Удаляем пробелы
                            views=0
                        ))

                if news_items:
                    session.add_all(news_items)
                    session.commit()
                    print(f"Successfully loaded {len(news_items)} news items.")
                else:
                    print("data.csv is empty.")

            except FileNotFoundError:
                print("Error: data.csv file not found. Skipping news loading.")
            except Exception as e:
                print(f"Error loading news from CSV: {e}")
        else:
            print(f"News articles already exist ({len(news_count)}). Skipping news loading.")


if __name__ == "__main__":
    load_data()