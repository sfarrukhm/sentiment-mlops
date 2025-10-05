# Use flake8 or pylint or ruff — here we’ll use ruff (fast modern linter)
# You can switch to flake8 or pylint if you prefer.

# The default target when you just run `make`
.PHONY: lint
lint:
	@echo "🔍 Linting all Python files..."
	@find . -type f -name "*.py" | xargs ruff check
	@echo "✅ Linting complete!"

# Optional: autofix formatting issues
.PHONY: lint-fix
lint-fix:
	@echo "🧹 Auto-fixing lint issues..."
	@find . -type f -name "*.py" | xargs ruff check --fix
	@echo "✨ Auto-fix complete!"
