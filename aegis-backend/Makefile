.PHONY: schemas test

schemas:
	@echo "Exporting protocol schemas to frontend..."
	python3 -m src.ws.protocol --export-schemas --output-dir ../frontend/src/ws/schemas
	@echo "Schemas updated in ../frontend/src/ws/schemas"

test:
	pytest tests/ -v
