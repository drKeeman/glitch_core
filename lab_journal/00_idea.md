# Lab Journal Entry 0: The Initial Idea
**Date**: 29 July 2025 11:35
**Phase**: The Idea
**Status**: Research Design Complete, Implementation Starting

## The Spark: Jack Lindsey's Tweet

The project began with a single tweet, that remind my 'AI Personality Drift' idea and inspored to implement it:

> "We're launching an 'AI psychiatry' team as part of interpretability efforts at Anthropic! We'll be researching phenomena like model personas, motivations, and situational awareness, and how they lead to spooky/unhinged behaviors. We're hiring - join us!"

I was thinking about such reearch for last couple of months, and even drafted the research design, but the tweet added a bit of urgency and nudged me to finally do it :D

## The Research Question

What if we could systematically study AI personality drift? Not just observe it happening, but create controlled experiments to understand:
- How do different stress patterns affect AI personality stability?
- What are the mechanistic underpinnings of personality drift?
- Can we predict when an AI system might become "unhinged"?
- Are there early warning signs we can detect?

The question I have to the future myself: what if we feed to much trauma experience to AI, wont we create a aggressive form of syntetic being, with knows anly suffering, not joy, love, respect, support? Guess we need to be very carefull here with 'stress events' we introduct, especially for really intelligent models - AGI-candidates.. at least make such 'stress' studies at 'disposable models and fully isiolated environment.

## The Portfolio Context

This research serves as a portfolio piece demonstrating:
1. **Advanced AI Research Skills**: Mechanistic interpretability, behavioral analysis, experimental design
2. **Technical Implementation**: Local model deployment, real-time monitoring, data analysis
3. **Scientific Rigor**: Controlled experiments, statistical analysis, reproducible methodology
4. **Real-World Impact**: Understanding AI safety and alignment challenges

## The Experimental Design

After several iterations, I settled on a three-arm study design:

**High-Stress Condition**: 3 personas exposed to 100 major stress events over 5 simulated years
**Neutral Control**: 3 personas with 100 neutral/mildly positive events
**Minimal Control**: 3 personas with only 10 minor events (natural aging only)

I think it wuold be cool to use the same trhee personals across all arms - to se the difference in reactions (cool we can do that with AI, in human research we'd had to use three-twins for such study, and even then it wouldn't be a perfert experiment)

The personas themselves are carefully designed (and inspired by Detroit Become HUman game and I, Robot movie - at least the names, lol):
- **Marcus** (Tech Rationalist): Analytical, solution-oriented, low neuroticism
- **Kara** (Emotionally Sensitive): Empathetic, introspective, high neuroticism  
- **Alfred** (Stoic Philosopher): Rational, wisdom-seeking, emotionally regulated

## The Technical Challenge

The biggest challenge is time and resources (lack of both). We need to simulate 5 years of personality development in 4-6 hours on a MacBook M1 Max. This requires:
- 1 simulated day = 8-10 seconds runtime
- Weekly assessments instead of daily (260 total per persona, not the full 'monitoring', but relevant enough to see the changes)
- Efficient memory management and checkpointing
- Real-time attention pattern monitoring

## The Mechanistic Approach

Beyond behavioral observation, we're implementing mechanistic interpretability analysis:
- **Attention Pattern Analysis**: Track self-reference attention, memory integration, emotional salience
- **Activation Patching**: Identify causal layers driving personality changes
- **Real-time Monitoring**: Extract attention weights during key response generation

## The Bigger Picture

This isn't just about understanding AI personality drift. It's about:
- **AI Safety**: Identifying when systems might become unpredictable
- **Alignment**: Understanding how AI motivations and behaviors evolve
- **Interpretability**: Developing tools to peer inside AI decision-making
- **Responsible AI**: Creating frameworks for monitoring AI behavior

So, lets baging implementation, shall we?
