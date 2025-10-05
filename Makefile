# Lint all Python files recursively with pylint

.PHONY: lint
lint:
	@echo "🔍 Linting all Python files with pylint..."
	@find . -type f -name "*.py" | xargs pylint --disable=R,C
	@echo "✅ Linting complete!"

