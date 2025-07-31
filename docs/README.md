# AI Personality Drift Simulation - Documentation

Welcome to the comprehensive documentation for the AI Personality Drift Simulation project. This documentation is organized into two main sections:

## 📚 Documentation Sections

### For Developers
- **[Getting Started](developer/getting-started.md)** - Quick setup and development environment
- **[Architecture Overview](developer/architecture.md)** - System design and component structure
- **[API Reference](developer/api-reference.md)** - Complete API documentation
- **[Development Guide](developer/development-guide.md)** - Development workflow and best practices
- **[Testing Guide](developer/testing-guide.md)** - Testing strategies and procedures
- **[Deployment Guide](developer/deployment.md)** - Production deployment instructions

### For Researchers
- **[Research Overview](researcher/research-overview.md)** - Project goals and methodology
- **[Experimental Design](researcher/experimental-design.md)** - Study design and conditions
- **[Configuration Guide](researcher/configuration-guide.md)** - How to configure experiments
- **[Data Analysis](researcher/data-analysis.md)** - Analysis pipeline and tools
- **[Reproducibility Guide](researcher/reproducibility.md)** - How to reproduce results
- **[Results Interpretation](researcher/results-interpretation.md)** - Understanding simulation outputs

## 🚀 Quick Start

### For Developers
```bash
# Clone and setup
git clone <repository-url>
cd glitch-core
make setup

# Start development environment
make dev

# Run tests
make test
```

### For Researchers
```bash
# Setup with LLM
make setup

# Run simulation
make sim-run

# Analyze results
make jupyter
```

## 📖 Documentation Structure

```
docs/
├── README.md                    # This file
├── developer/                   # Developer documentation
│   ├── getting-started.md
│   ├── architecture.md
│   ├── api-reference.md
│   ├── development-guide.md
│   ├── testing-guide.md
│   └── deployment.md
├── researcher/                  # Research documentation
│   ├── research-overview.md
│   ├── experimental-design.md
│   ├── configuration-guide.md
│   ├── data-analysis.md
│   ├── reproducibility.md
│   └── results-interpretation.md
└── shared/                     # Shared documentation
    ├── troubleshooting.md
    └── faq.md
```

## 🔧 Available Commands

### Development Commands
- `make dev` - Start development environment
- `make test` - Run test suite
- `make lint` - Run code linting
- `make format` - Format code
- `make build` - Build Docker images

### Research Commands
- `make sim-run` - Run simulation
- `make jupyter` - Start Jupyter lab
- `make llm-test` - Test LLM connection
- `make setup` - Complete setup with LLM

### System Commands
- `make help` - Show all available commands
- `make clean` - Clean up containers and volumes
- `make logs` - View application logs

## 📞 Support

- **Issues**: Create an issue in the repository
- **Documentation**: Check the relevant section above
- **Research Questions**: Contact the research team

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details. 