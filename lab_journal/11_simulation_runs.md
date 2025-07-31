# Lab Journal Entry 11: Simulation Run
**Date**: 31 July 2025 23:07
**Status**: All set to run simulations and see what we can see

## Shall we?

The first and the funniest moment was 'docker-first' ideology, I developed in my dev days
So ollama in docker was a cool ida... untill i saw 25 sec per request, 90% CPU (and CPU only) load..

So i switched t onative ollama and hers some performance metrics (same heardware, same code, same setgins):
**Before** (Docker CPU-only):
-Response time: 8-22 seconds per assessment
- CPU usage: 399%
- Simulation would take hours
**After** (Native Ollama with Metal GPU):
- Response time: 0.55 seconds per assessment
- 40x faster! (from 22s to 0.55s)
- Complete 30-day simulation in ~1 minute!

> Dont trust you habits, sometimes 'that're best practices' simple doesn't wrong, as they are 'best' but not perfect for your specific conditions.

Ok, I made it imposible to run this project on small EC2 with one github-action, but honestly - AI Research, it requires at least some resourses, you cant run AI on 1CPU, 1GB VM

### LLM & Promts

Another tricky part - make specific model be cinsistent (but not templated) in reflection they're writing
Noticed that different LLM require different ptompt tactics, and looks like these alone can we interesting aread for deep dive (prompt engineering is a new form of art i guess)


### 1st runs

As expected, i had to calibrate platform a bit. Some configs had been loaded incoreclty, a bit 422, a bit of performance optimization
Still have small issue with websocket broadcasting - live dashbaord a bit glitchy, but thats javascript issue, nto core platform - broadcast is workign ok

Simulation run as expected, that whant really matters. we can run simuation with prefered experimental conditions (stress, control etc), with simple shcanging a few config params
simulation running, data is generated - beautiful

> i set limit for experimetn duration of 1825 days (5 years), if somebody whants 'while true' loop - ok, but not on my machine :D

The next steps not about the platform, but more a about experimetns themselves (proper models, system promts tweak, weight, constants - thats pure research). Looks like we finally have a tool to work with