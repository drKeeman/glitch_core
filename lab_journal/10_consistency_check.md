# Lab Journal Entry 10: Sanity Check
**Date**: 31 July 2025 16:05
**Status**: All variables chekced and moved to yaml configs. No hardoced staff inside code. PLus decouipled 'Empiric Variables' into dedicated configs - i feel we would bed to 'play' with them a lot


## Follow the white rabbit

As I was in rush-mode, building this project, I hardcided some staff for speed-sake, but its time to check and remove all this hardcoded BS, so we can fully rely on single source-of-truth and no one would suffer in future blindly guessing 'why my condig is not fully applied'

> Perfosnal reflection: it's ok to hardcode things while you prototyping, but the one golden rule - replace hardcoded values to dynamic vars at the same iteration, otherwise in a couple of weeks - a months, you'd just forget about them and here comes the tech-debpt, bugs and 'itallian-driven design' (read as 'php spaghetty')


The targets to check:
- personas ✅
- events ✅
- experimental_design ✅

The targets to think about - coefficients we use (like severioty of event and its weight) - 1st simulations raised questions here (what value to use - that would be emperic guessing), so better to make them dynamic (this alone can we a separate study) and either as part of experiment design, or maybe even separate config. Lets see..

So what else I move to yaml configs:
1. **`config/experiments/clinical_thresholds.yaml`**
   - PHQ-9, GAD-7, PSS-10 severity thresholds
   - Clinical significance thresholds
   - Risk assessment criteria

2. **`config/experiments/drift_detection.yaml`**
   - Drift detection thresholds
   - Baseline sample requirements
   - Early warning parameters

3. **`config/experiments/personality_drift.yaml`**
   - Stress level scaling factors
   - Trauma level coefficients
   - Activation sparsity thresholds

4. **`config/experiments/simulation_timing.yaml`**
   - Duration and compression factors
   - Assessment intervals
   - Checkpoint frequencies

5. **`config/experiments/mechanistic_analysis.yaml`**
   - Attention sampling rates
   - Event intensity ranges
   - Circuit tracking parameters


'Great Sucess' as Borat said - we even dindt broke the test (almost), so now we have full yanl-configurable platform, and it would be pretty easy to setup experimetns with various params (another cool part, we always know what params we use for simulation, and track studies - jsut remember to take a snapshot of configs you using during sim run).

Idea for future - add config exports alongside with sim run and result exports, so we can consistently work with proper experiment version, But its a TODO - for now, a bit of 'memorizing' and 'attention' should do the job