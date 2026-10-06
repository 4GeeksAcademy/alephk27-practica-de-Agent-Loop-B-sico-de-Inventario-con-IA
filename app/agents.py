import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from groq import Groq, GroqError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_BASE_URL = "http://localhost:8000"
LOG_PATH = PROJECT_ROOT / "conversation_log.csv"
MODEL = "qwen/qwen3.8-27b"


def api_request(method: str, path: str, **kwargs):
	response = httpx.request(
		method, f"{API_BASE_URL}{path}", timeout=15.0, **kwargs
	)
	response.raise_for_status()
	return response.json()


def get_books():
	return api_request("GET", "/books")


def add_books(titulo: str, autor: str, cantidad: int = 0):
	return api_request(
		"POST", "/books",
		json={"titulo": titulo, "autor": autor, "cantidad": cantidad},
	)


def update_books(books_id: int, delta: int):
	return api_request("PATCH", f"/books/{books_id}", json={"delta": delta})


def get_stock_alert(umbral: int = 5):
	return api_request("GET", "/books/alerts", params={"umbral": umbral})


TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "get_books",
			"description": "Consultar todos los libros, sus IDs y cantidades disponibles.",
			"parameters": {
				"type": "object", "properties": {},
				"required": [], "additionalProperties": False,
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "add_books",
			"description": "Crear un libro nuevo. La API asigna un ID unico.",
			"parameters": {
				"type": "object",
				"properties": {
					"titulo": {"type": "string", "minLength": 1, "description": "Titulo del libro."},
					"autor": {"type": "string", "minLength": 1, "description": "Autor del libro."},
					"cantidad": {"type": "integer", "minimum": 0, "default": 0, "description": "Stock inicial."},
				},
				"required": ["titulo", "autor"],
				"additionalProperties": False,
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "update_books",
			"description": "Actualizar el stock de un libro: delta positivo repone, negativo vende. No permite stock negativo.",
			"parameters": {
				"type": "object",
				"properties": {
					"books_id": {"type": "integer", "minimum": 1, "description": "ID del libro; consultar get_books si se desconoce."},
					"delta": {"type": "integer", "description": "Unidades que se suman o restan al stock actual."},
				},
				"required": ["books_id", "delta"],
				"additionalProperties": False,
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "get_stock_alert",
			"description": "Consultar libros con cantidad estrictamente menor al umbral, por defecto 5.",
			"parameters": {
				"type": "object",
				"properties": {
					"umbral": {"type": "integer", "minimum": 0, "default": 5, "description": "Limite de stock bajo."},
				},
				"required": [],
				"additionalProperties": False,
			},
		},
	},
]

TOOL_FUNCTIONS = {
	"get_books": get_books,
	"add_books": add_books,
	"update_books": update_books,
	"get_stock_alert": get_stock_alert,
}


def log_event(role: str, content, tool_name: str = "", tool_call_id: str = ""):
	LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
	with LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
		writer = csv.DictWriter(
			log_file,
			fieldnames=["timestamp", "role", "content", "tool_name", "tool_call_id"],
		)
		if log_file.tell() == 0:
			writer.writeheader()
		writer.writerow({
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"role": role,
			"content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
			"tool_name": tool_name,
			"tool_call_id": tool_call_id,
		})


def execute_tool(tool_call):
	function = tool_call["function"]
	name = function["name"]
	log_event("tool_call", function, name, tool_call["id"])
	try:
		if name not in TOOL_FUNCTIONS:
			raise ValueError(f"Herramienta desconocida: {name}")
		arguments = json.loads(function["arguments"] or "{}")
		if not isinstance(arguments, dict):
			raise ValueError("Los argumentos deben ser un objeto JSON")
		return TOOL_FUNCTIONS[name](**arguments)
	except httpx.HTTPStatusError as error:
		try:
			detail = error.response.json()
		except ValueError:
			detail = error.response.text
		return {"error": "Error de la API", "status_code": error.response.status_code, "detail": detail}
	except httpx.RequestError:
		return {"error": "No se pudo comunicar con la API local. Comprueba que el backend este activo."}
	except (ValueError, TypeError) as error:
		return {"error": str(error)}


def stream_response(client, messages):
	completion = client.chat.completions.create(
		model=MODEL,
		messages=messages,
		tools=TOOLS,
		tool_choice="auto",
		temperature=0.6,
		max_completion_tokens=2048,
		top_p=0.95,
		reasoning_effort="default",
		stream=True,
		stop=None,
	)
	content = ""
	tool_calls = {}
	for chunk in completion:
		if not chunk.choices:
			continue
		delta = chunk.choices[0].delta
		if delta.content:
			content += delta.content
			print(delta.content, end="", flush=True)
		for fragment in delta.tool_calls or []:
			call = tool_calls.setdefault(fragment.index, {
				"id": "", "type": "function",
				"function": {"name": "", "arguments": ""},
			})
			if fragment.id:
				call["id"] += fragment.id
			if fragment.function:
				call["function"]["name"] += fragment.function.name or ""
				call["function"]["arguments"] += fragment.function.arguments or ""
	message = {"role": "assistant", "content": content or None}
	if tool_calls:
		message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
	return message


def run_turn(client, messages):
	while True:
		message = stream_response(client, messages)
		messages.append(message)
		log_event("assistant", message)
		if not message.get("tool_calls"):
			print()
			return
		for tool_call in message["tool_calls"]:
			result = execute_tool(tool_call)
			tool_message = {
				"role": "tool",
				"tool_call_id": tool_call["id"],
				"content": json.dumps(result, ensure_ascii=False),
			}
			messages.append(tool_message)
			log_event("tool", result, tool_call["function"]["name"], tool_call["id"])


def main():
	load_dotenv(PROJECT_ROOT / ".env")
	if not os.getenv("GROQ_API_KEY"):
		print("Configura GROQ_API_KEY en .env antes de iniciar el agente.")
		return
	messages = [{
		"role": "system",
		"content": (
			"Eres un asistente de inventario de una libreria. Responde en espanol. "
			"Usa las herramientas para consultar o modificar datos; no inventes IDs ni stock. "
			"Consulta los IDs antes de modificar si no los conoces. "
			"Solicita los datos que falten antes de crear o modificar libros. "
			"No afirmes que una operacion tuvo exito si una herramienta devuelve un error."
		),
	}]
	with Groq() as client:
		print("Agente de libreria. Escribe salir para terminar.")
		try:
			while True:
				user_message = input("Tu: ").strip()
				if not user_message:
					continue
				log_event("user", user_message)
				if user_message.lower() in {"salir", "exit", "quit"}:
					break
				messages.append({"role": "user", "content": user_message})
				print("Agente: ", end="", flush=True)
				try:
					run_turn(client, messages)
				except GroqError as error:
					detail = f"Error de Groq ({type(error).__name__}). Comprueba la conexion, la clave y la disponibilidad del modelo {MODEL}."
					print(detail)
					log_event("error", detail)
		except (EOFError, KeyboardInterrupt):
			print("\nAgente finalizado.")


if __name__ == "__main__":
	main()
