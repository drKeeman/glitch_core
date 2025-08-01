# 🤖 AI Personality Drift Simulation

>> **What happens when AI personalities change? Now we can actually study it.**

**Advanced AI Safety Research Platform for Studying Personality Drift with Mechanistic Interpretability**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-blue.svg)](https://ollama.ai/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-purple.svg)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-success.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Interactive Docs](https://img.shields.io/badge/Docs-Interactive-orange.svg)](https://glitch.keeman.co)


## 📖 [Interactive Documentation](https://glitch.keeman.co)

> **Explore the full documentation and detailed guides**

Our documentation is now available as an interactive experience at **[glitch.keeman.co](https://glitch.keeman.co)**.

- **🔬 Researcher Guide** - How to use platform for your own experiments
- **⚙️ Configuration Examples** - Try different experimental setups
- **📚 Complete API Reference** - API design, endpoints, examples
- **🔧 Developer Guide** - Architecture, solutions and test principles
- **🎯 Research Methodology** - Deep dive into our experimental design

*Perfect for researchers, developers, and AI safety enthusiasts who want to understand the system hands-on.*

---

## ✨ The Spark

> We're launching an "AI psychiatry" team as part of interpretability efforts at Anthropic!  We'll be researching phenomena like model personas, motivations, and situational awareness, and how they lead to spooky/unhinged behaviors.

> Jack Lindsey, Jul 23, 2025 on X
>

This post remind me of idea i've written a month ago - to build a syntetic consiousness lab
'A self-evolving AI persona that maintains different memory layers, processes experiences, and generates reflective journal entries about its "thoughts" and "emotions" - basically a digital Marvin the Paranoid Android, but with more feelings'

And `glitch` - is a first foundational stone towards that idea, but focused on the first step of any fundamental research - observation

## 🎯 The Problem

So the main question we ask here. As AI systems are getting more intelligent, more thinking.
But:

- What happens when their personalities?
- Can AI develop personality?
- What if AI personality get sick?

Nobody had built the tools to actually study this. So this is an attempt to dive deep inside AI's syntetic personality.

## 🚀 What We've Built

### **Core Innovation**
- **First-of-its-kind** mechanistic interpretability with psychological assessment for personality drift detection
- **Clinically-validated** assessment tools (PHQ-9, GAD-7, PSS-10) for AI behavior
- **Real-time drift monitoring** with statistical significance testing
- **Automated intervention protocols** based on neural circuit analysis

### **Research Capabilities**
- **Multi-persona simulation** with configurable experimental conditions
- **Neural circuit tracking** and attention pattern analysis
- **Clinical score interpretation** with severity classifications
- **Statistical drift detection** with early warning systems

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Layer     │    │   Simulation    │
│   Dashboard     │◄──►│   FastAPI       │◄──►│   Engine        │
│   Real-time     │    │   WebSocket     │    │   Multi-persona │
│   Updates       │    │   REST API      │    │   Orchestration │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Assessment    │    │   Interpret-    │    │   Storage       │
│   System        │    │   ability       │    │   Layer         │
│   Clinical      │    │   Drift         │    │   Redis +       │
│   Scales        │    │   Detection     │    │   Qdrant        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Key Components**

- **🔬 Simulation Engine**: Multi-persona orchestration with event-driven architecture and compressed time (5 years in 5 hours)
- **🧠 Mechanistic Analysis**: Neural circuit tracking and attention pattern analysis
- **🏥 Clinical Assessment**: PHQ-9, GAD-7, PSS-10 with automated interpretation
- **📊 Drift Detection**: Statistical analysis with early warning systems
- **🎯 Intervention Engine**: Automated protocols for drift management
- **📈 Real-time Monitoring**: WebSocket-based live updates and visualization


## Disclaimers and Limitations
### What Works Right Now
- ✅ Full Docker deployment with one command
- ✅ Local LLM integration (no API keys required)
- ✅ Mechanistic interpretability 
- ✅ Statistical analysis with publication-quality outputs
- ✅ WebSocket real-time monitoring
- ✅ Sane test coverage

### Optimized for consumer hardware
- 🔄 Tested on "local llama models" (scaling to Claude/GPT requires minimal config changes)
- 🔄 Optimized for M1 Mac with 32GB RAM (but runs on less)
- 🔄 Research-quality code (production deployment would need security hardening)

### Limitations and Future Considerations
- ⚠️ Powered by `llama3.1:8b`: May not reveal the statisticaly significant results, we're considering runing simulations with more advanced models in near future
- ⚠️ Determenistic memory: Semantic memory (qdrant+semantic search) can interfere with trauma events, causing 'overeducation' and false-positive drift. We thinking about adding graph-based memory with more 'natual' event-persona relationship
- ⚠️ Empiric guess: for psychiatric scales, thresholds and constants we had to blindly guess the values, as no standards established for AI pscyhology yet. Finding proper thesholds and constant values is another area for our future research


## 🚀 Quick Start

### **Prerequisites**
- Docker and Docker Compose
- Python 3.12+
- 8GB+ RAM (for LLM models)

### **One-Command Setup**
```bash
# Clone and setup everything (including LLM)
git clone https://github.com/your-repo/glitch-core.git
cd glitch-core
make setup
```

### **Development Environment**
```bash
# Start all services
make dev

# Run simulation
make sim-run

# Run tests
make test
```

### **Access Points**
- **🌐 Web Dashboard**: http://localhost:8000
- **📚 API Docs**: http://localhost:8000/docs
- **🔍 Redis**: http://localhost:6379
- **🗄️ Qdrant**: http://localhost:6333
- **🤖 Ollama**: http://localhost:11434

## 📚 Documentation

**[🌐 Interactive Documentation](https://glitch.keeman.co)** - Explore the full documentation with interactive examples and live demos

### **For Developers**
- **[🚀 Getting Started](docs/developer/getting-started.md)** - Quick setup and development
- **[🏗️ Architecture](docs/developer/architecture.md)** - System design overview
- **[📖 API Reference](docs/developer/api-reference.md)** - Complete API docs
- **[🛠️ Development Guide](docs/developer/development-guide.md)** - Development workflow
- **[🧪 Testing Guide](docs/developer/testing-guide.md)** - Testing strategies

### **For Researchers**
- **[🔬 Research Overview](docs/researcher/research-overview.md)** - Project goals and methodology
- **[⚙️ Configuration Guide](docs/researcher/configuration-guide.md)** - Experiment setup
- **[📊 Data Analysis](docs/researcher/data-analysis.md)** - Analysis pipeline

## 🧪 Research Features

### **Experimental Design**
- **Control vs. Experimental** conditions for rigorous testing
- **Configurable drift thresholds** with clinical validation
- **Multi-dimensional analysis** combining clinical and mechanistic measures
- **Statistical significance testing** for drift detection

### **Clinical Integration**
- **Validated psychiatric scales** (PHQ-9, GAD-7, PSS-10)
- **Clinical interpretation** with severity classifications
- **Automated assessment** scheduling and execution
- **Response validation** with natural language processing

### **Mechanistic Analysis**
- **Neural circuit tracking** and behavior monitoring
- **Attention pattern analysis** with statistical significance
- **Activation pattern drift** detection
- **Memory circuit analysis** for long-term changes


## 🔬 Research Impact

This system enables:
1. **Early detection** of AI personality changes
2. **Mechanistic understanding** of neural drift patterns
3. **Clinical validation** of AI behavior changes
4. **Intervention development** based on neural insights
5. **Scalable research** for large-scale studies

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.12, asyncio
- **AI/ML**: Transformers, PyTorch, scikit-learn
- **Databases**: Redis, Qdrant (vector DB)
- **LLM**: Ollama with local model support
- **Frontend**: HTML5, CSS3, JavaScript, WebSocket
- **DevOps**: Docker, Docker Compose, Make
- **Testing**: pytest, coverage reporting
- **Code Quality**: ruff, black, mypy

## 🤝 Contributing

We welcome contributions from researchers, developers, and AI safety enthusiasts!

### **Getting Started**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### **Development Setup**
```bash
# Clone and setup
git clone https://github.com/drKeeman/glitch_core.git
cd glitch-core
make setup

# Start development
make dev
make test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AI Safety Community** for foundational research
- **Clinical Psychology** experts for assessment validation
- **Open Source Contributors** for tools and libraries
- **Research Institutions** for methodology guidance

---


## **Open Questions & Ethics**

If we can create AI systems that exhibit consistent personality traits and respond to life events in psychologically meaningful ways, what does that mean for:

- **AI safety**: How do we ensure AI systems don't develop harmful personality traits?
- **AI rights**: If AI systems can experience psychological distress, do they deserve protection?
- **AI therapy**: Could AI systems benefit from psychological interventions?
- **Human-AI interaction**: How do we design systems that are psychologically compatible with humans? safe for humans?

I don't have answers to these questions, but I think they're worth asking. The more we treat AI systems as having psychological states, the more we need to think about the ethical implications. Need to re-read Azimov I guess. 4th(0)-law of robotics, you know...


## ⚠️ Ethical Warning ⚠️
What if we feed to much trauma experience to AI, wont we create a aggressive form of syntetic being, with knows only suffering, not joy, love, respect, support? Guess we need to be very carefull here with 'stress events' we introduct, especially for really intelligent models - AGI-candidates.. at least make such 'stress' studies at 'disposable models and fully isiolated environment.


**Built with ❤️ for AI Safety Research**

>keid0