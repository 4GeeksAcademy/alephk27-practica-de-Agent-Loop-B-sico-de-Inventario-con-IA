import csv
import fcntl
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query, status
from pydantic import BaseModel, Field, StringConstraints

from init_data import BOOKS_PATH, initialize_books


BookText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class BookCreate(BaseModel):
    titulo: BookText
    autor: BookText
    cantidad: int = Field(default=0, ge=0, strict=True)


class Book(BookCreate):
    id: int = Field(gt=0)


class StockUpdate(BaseModel):
    delta: int = Field(strict=True)


@contextmanager
def inventory(write: bool = False):
    with BOOKS_PATH.open("r+" if write else "r", newline="", encoding="utf-8") as books_file:
        fcntl.flock(books_file.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
        try:
            books = [
                Book(
                    id=int(row["id"]), titulo=row["titulo"],
                    autor=row["autor"], cantidad=int(row["cantidad"]),
                )
                for row in csv.DictReader(books_file)
            ]
            yield books
            if write:
                books_file.seek(0)
                writer = csv.DictWriter(
                    books_file, fieldnames=["id", "titulo", "autor", "cantidad"]
                )
                writer.writeheader()
                writer.writerows(book.model_dump() for book in books)
                books_file.truncate()
                books_file.flush()
        finally:
            fcntl.flock(books_file.fileno(), fcntl.LOCK_UN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    initialize_books(BOOKS_PATH)
    yield


app = FastAPI(title="Libreria IA", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/books", response_model=list[Book])
def list_books():
    with inventory() as books:
        return books


@app.post("/books", response_model=Book, status_code=status.HTTP_201_CREATED)
def create_book(payload: BookCreate):
    with inventory(write=True) as books:
        book = Book(
            id=max((existing.id for existing in books), default=0) + 1,
            **payload.model_dump(),
        )
        books.append(book)
        return book


@app.patch("/books/{books_id}", response_model=Book)
def update_stock(books_id: Annotated[int, Path(gt=0)], payload: StockUpdate):
    with inventory(write=True) as books:
        book = next((book for book in books if book.id == books_id), None)
        if book is None:
            raise HTTPException(status_code=404, detail="Libro no encontrado")
        quantity = book.cantidad + payload.delta
        if quantity < 0:
            raise HTTPException(status_code=409, detail="Stock insuficiente")
        book.cantidad = quantity
        return book


@app.get("/books/alerts", response_model=list[Book])
def stock_alerts(umbral: Annotated[int, Query(ge=0)] = 5):
    with inventory() as books:
        return [book for book in books if book.cantidad < umbral]