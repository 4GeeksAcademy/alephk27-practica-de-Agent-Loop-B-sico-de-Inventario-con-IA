import csv
from pathlib import Path


BOOKS_PATH = Path(__file__).resolve().parent / "data" / "books.csv"


def initialize_books(path: Path = BOOKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", newline="", encoding="utf-8") as books_file:
            writer = csv.DictWriter(
                books_file, fieldnames=["id", "titulo", "autor", "cantidad"]
            )
            writer.writeheader()
            writer.writerows([
                {"id": 1, "titulo": "Don Quijote", "autor": "Miguel de Cervantes", "cantidad": 5},
                {"id": 2, "titulo": "Cien anos de soledad", "autor": "Gabriel Garcia Marquez", "cantidad": 3},
            ])
    except FileExistsError:
        pass


if __name__ == "__main__":
    initialize_books()