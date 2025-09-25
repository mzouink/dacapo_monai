# Makefile for DaCapo-MONAI documentation

# You can set these variables from the command line, and also from the environment
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Default target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help clean html pdf linkcheck livehtml install deps

# Clean build directory
clean:
	@echo "🧹 Cleaning build directory..."
	rm -rf $(BUILDDIR)/*
	@echo "✅ Clean completed!"

# Install documentation dependencies
install:
	@echo "📦 Installing documentation dependencies..."
	pip install -e "../[docs]"
	@echo "✅ Dependencies installed!"

# Install just docs dependencies
deps:
	@echo "📦 Installing Sphinx and extensions..."
	pip install sphinx sphinx-design furo myst-parser sphinx-copybutton sphinxcontrib-mermaid sphinx-autobuild
	@echo "✅ Documentation dependencies installed!"

# Build HTML documentation
html: deps
	@echo "🔨 Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "✅ HTML documentation built successfully!"
	@echo "📖 Open $(BUILDDIR)/html/index.html in your browser"

# Build PDF documentation
pdf: deps
	@echo "🔨 Building PDF documentation..."
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	@echo "📄 LaTeX files generated, building PDF..."
	@cd $(BUILDDIR)/latex && make all-pdf
	@echo "✅ PDF documentation built successfully!"
	@echo "📄 PDF available at $(BUILDDIR)/latex/dacapo-monai.pdf"

# Check for broken links
linkcheck: deps
	@echo "🔗 Checking for broken links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
	@echo "✅ Link check completed!"

# Live reload development server
livehtml: deps
	@echo "🚀 Starting live documentation server..."
	@echo "📖 Documentation will be available at http://localhost:8000"
	@echo "💡 Press Ctrl+C to stop the server"
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --host 0.0.0.0 --port 8000

# Build all formats
all: clean html pdf
	@echo "🎉 All documentation formats built successfully!"

# Quick build (HTML only, no dependency check)
quick:
	@echo "⚡ Quick HTML build..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "✅ Quick build completed!"

# Serve built documentation
serve:
	@echo "🌐 Serving documentation at http://localhost:8080"
	@echo "💡 Press Ctrl+C to stop the server"
	@cd $(BUILDDIR)/html && python -m http.server 8080

# Development workflow
dev: clean livehtml

# CI workflow  
ci: clean html linkcheck
	@echo "🤖 CI documentation build completed!"

# Generate API documentation
api:
	@echo "🔧 Generating API documentation..."
	sphinx-apidoc -o api_reference ../src/dacapo_monai --force --separate
	@echo "✅ API documentation generated!"

# Spell check (requires aspell)
spellcheck:
	@echo "🔤 Spell checking documentation..."
	@find . -name "*.md" -exec aspell --mode=sgml --check {} \; || echo "⚠️ aspell not installed, skipping spell check"

# Statistics
stats:
	@echo "📊 Documentation Statistics:"
	@echo "📄 Markdown files: $$(find . -name '*.md' | wc -l)"
	@echo "📄 reStructuredText files: $$(find . -name '*.rst' | wc -l)"
	@echo "📄 Total lines: $$(find . -name '*.md' -o -name '*.rst' | xargs wc -l | tail -1)"
	@if [ -d "$(BUILDDIR)/html" ]; then \
		echo "🌐 Generated HTML files: $$(find $(BUILDDIR)/html -name '*.html' | wc -l)"; \
		echo "📦 Build size: $$(du -sh $(BUILDDIR)/html | cut -f1)"; \
	fi

# Help with common tasks
info:
	@echo "📚 DaCapo-MONAI Documentation Build System"
	@echo ""
	@echo "📋 Common Commands:"
	@echo "  make install     - Install documentation dependencies"
	@echo "  make html        - Build HTML documentation"
	@echo "  make livehtml    - Start live reload server for development"  
	@echo "  make pdf         - Build PDF documentation"
	@echo "  make linkcheck   - Check for broken links"
	@echo "  make clean       - Clean build directory"
	@echo "  make serve       - Serve built documentation"
	@echo "  make stats       - Show documentation statistics"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make dev         - Clean build and start live server"
	@echo "  make quick       - Quick HTML build without dependency check"
	@echo "  make api         - Generate API documentation"
	@echo ""
	@echo "🤖 CI/CD:"
	@echo "  make ci          - Full CI build with link checking"
	@echo ""
	@echo "💡 For more options, run 'make help'"

# Default target when no specific target is provided
.DEFAULT_GOAL := info