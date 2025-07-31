# Lab Journal Entry 10: Sanity Check
**Date**: 31 July 2025 13:31
**Status**: Main app is done, final polish ongoing


## Follow the white rabbit

As I was in rush-mode, building this project, I hardcided some staff for speed-sake, but its time to check and remove all this hardcoded BS, so we can fully rely on single source-of-truth and no one would suffer in future blindly guessing 'why my condig is not fully applied'

> Perfosnal reflection: it's ok to hardcode things while you prototyping, but the one golden rule - replace hardcoded values to dynamic vars at the same iteration, otherwise in a couple of weeks - a months, you'd just forget about them and here comes the tech-debpt, bugs and 'itallian-driven design' (read as 'php spaghetty')


The targets to check:
- personas
- events
- experimental_design

The targets to think about - coefficients we use (like severioty of event and its weight) - 1st simulations raised questions here (what value to use - that would be emperic guessing), so better to make them dynamic (this alone can we a separate study) and either as part of experiment design, or maybe even separate config. Lets see..

