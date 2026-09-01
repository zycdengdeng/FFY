# Speaker Script — Week 5 International Meeting
**Generative Design of Bio-Inspired Landing Gear** · Zihan Wang · 2026-09-09
Audience: Prof. K. Jimmy Hsia · Prof. Yifan Zhao · **Prof. Davide Bigoni**

> Target: ~18 min talk + discussion. Lines marked **[Q]** are questions to put to the room.
> Lines marked **⚠ STALE** flag where the slide no longer matches our current results —
> say the correction out loud, or update the slide first (see the note at the end).

---

## S1 · Title

Good morning, and thank you for making the time. I am Zihan Wang, a joint PhD student between Shanghai Jiao Tong University and EIT, working with Professor Zhao. Today I want to walk you through a generative-design pipeline for bio-inspired landing gear — where the biology comes from, how it becomes a constraint the model must obey, and, honestly, where the approach is still thin. I would value your criticism more than your agreement.

## S2 · Background

Let me start with the motivation. Our original target was large fixed-wing aircraft — Airbus-class landing gear. That gear is built around oleo-pneumatic shock struts, designed to a safe-life philosophy. It is proven and reliable, but it is heavy, mechanically complex, and its damping is fixed at design time — one strut, one behaviour, every runway.

Our idea was to look at how large waterbirds land, and design a leg-like gear from that.

I should say up front that as the work progressed we came to think this pipeline fits **small fixed-wing UAVs** better than airliners. I will come back to why.

## S3 · System overview

In one sentence: a perception module reads foot motion from waterbird video; anatomical literature gives us the leg bone lengths; we turn those into physical priors; and a generative model then produces a landing gear for a specified operating condition.

The key word is *condition*. We are not designing one leg. We are designing a mapping from condition to leg.

## S4 · Architecture

Four blocks. The first is data processing — in current language, a data engine. This is where the biology and the physics constraints actually live.

Open data goes in on the left: waterbird video, anatomical datasets, landing-speed literature, geotechnical data. These become a mass-conditioned biological prior, the operating conditions, and the material constants.

Those feed the Exudyn physics simulator, which does the drop simulation. Results are screened and written into a database, which trains a conditional VAE. The model then proposes new designs, which are simulated again — that is the self-improving loop.

The output, bottom right: given aircraft mass, landing speed, terrain, and the limits on g-load and stroke, the model returns three bone lengths, three joint stiffnesses, and a damping ratio.

## S5 · Raw data

This is the foundation of the data engine, and every table on this slide shows real headers and real rows.

**AVONET** is a global bird database — we take its measured tarsometatarsus length. **EltonTraits** is a species-level trait database — we take body mass. Joined on standardised scientific name, that gives us 213 waterbird species.

**Watanabe 2017** originally used bone proportions to decide whether fossil ducks could fly; for us it is a source of measured lengths for all three hindlimb bones — femur, tibiotarsus, tarsometatarsus — across 91 volant Anatidae.

**Whitehead 2023** is high-speed video of mallards landing on water. From it we take our baseline touchdown speed, 1.2 metres per second.

The row at the bottom is what does *not* go in the tables — material constants, safety factor, the mass budget. Those are engineering practice, not biology, and I have kept them visibly separate.

## S6–S7 · The mass–leg-length fit

This is the central prior. Log leg length is linear in log body mass: an intercept a, an allometric exponent b, and a residual spread sigma. Across 213 species we get b = 0.391 with r-squared 0.75.

The exponent matters. Geometric similarity would give one third. Elastic similarity — McMahon's criterion — would give one quarter. We measure 0.391, above both. Waterbird legs get *relatively* longer as the bird gets heavier.

## S8 · Family-by-family — what we can and cannot say

Now the honest version of that fit.

**Only the Anatidae exponent is trustworthy.** 151 species spanning 1.6 decades of body mass gives b = 0.423 ± 0.034 — slightly steeper than the pooled 0.391.

The other four families cannot be compared, and the reason is on the slide: sample size and mass range. Pelicans have 8 species spanning 0.44 decades, and a confidence interval of ± 0.240 — that is no information at all. Loons look lowest at 0.225, but that is 5 points.

The error on a regression slope scales inversely with mass range times root n. A narrow range and a small sample is a double penalty. That is the statistical reason we did *not* build per-family priors.

But — and this is recent — the families *do* carry information. Just not in the slope. In the intercept.

On the right is u for each species: how long its leg is relative to the typical leg for that body mass. The family medians are Anatidae −0.43, cormorants +0.28, pelicans +0.75, loons +0.89, grebes +1.72.

That ordering is **strictly the same** as the ordering of how well each family walks on land. Ducks walk well and land on hard ground; grebes have legs set so far back that they cannot walk at all, and cannot take off from land.

So: the longer the leg relative to body mass, the less the bird comes ashore. My reading is that b describes how fast leg length grows *within* a family, and u describes whether that family is long-legged or short-legged overall. The first is limited by sample size; the second has a clear ecological meaning.

## S9–S10 · The two shape ratios

The previous slides fixed how long the leg is. These fix its *shape*.

We split the leg into three segments. r₂ is tibiotarsus over tarsometatarsus, r₃ is femur over tarsometatarsus. So L₁ sets absolute size; r₂ and r₃ set the shape.

The ranges come from Watanabe — the 91 volant species, full measured range: r₂ from 1.49 to 2.09, r₃ from 0.84 to 1.28. The mapping is linear: the network outputs a number between zero and one, mapped straight onto that interval. Note the difference from L₁ — the L₁ box shifts with body mass; the ratio boxes do not.

On the left, the 91 species in the r₂–r₃ plane; the red box is our design space. The orange diamond is the swan — our group's earlier baseline — sitting slightly below centre.

On the right is a check we ran: **are these ratios really size-independent?**

To avoid the spurious correlation you get when a ratio shares its denominator with the x-axis, we did not regress ratios. We regressed the raw segment lengths, log-log. The reference is clear: geometric similarity is an exponent of 1.0 — shape preserved.

We measure L₂ ∝ L₁^0.877 and L₃ ∝ L₁^0.766. Both significantly below one, with t-values of 7.7 and 11.8. **Double the tarsometatarsus and the tibiotarsus grows only 0.88-fold, the femur only 0.77-fold.**

So the size-independence assumption is flawed, and we detected it. The good news is that our band still covers the whole drift — the r₂ trend runs from 1.58 to 1.92, entirely inside 1.49 to 2.09. We exclude no real bird. The cost is only that the box is wider than strictly necessary.

## S11 · The prior formula

Putting it together: log L₁ = 0.479 + 0.391·log m + sigma·u_L, with L₁ in millimetres and m in grams.

sigma is 0.078 — the residual spread. u_L is produced by the network: it is *how much of an outlier this design is*, bounded at ±2.5, which covers 99.1 percent of the AVONET records.

That bound is a deliberate design choice, not a measurement. It is the leash: the model may propose an unusual bird, but not an impossible one.

## S12 · Joint stiffness — how the box was set

Joint stiffness is set by a dimensional rule: kappa, dimensionless, equals k over m·g·L_leg. The denominator is the gravitational moment the joint must at minimum hold. So kappa is a safety multiple, default 4.

The hip gets 4× that, from two independent sources: our group's earlier water-landing measurements, where the hip flexes substantially while knee and ankle stay steady; and our own structural sizing, which puts the peak hip moment 3–4× above knee and ankle.

**We do not preset a value — we preset a range**, roughly a factor of two either side of those defaults. What the model actually picks, it decides in training. After 41 self-improvement rounds it selects ankle 1.9–3.4, knee 3.6–5.6, hip 10.7–17.7. **None of the three sits at the centre of its box.** The box itself is fixed; training does not move it.

⚠ **STALE** — since this deck we widened the lower bounds (ankle to 0.75, hip to 3.0) because the model was pressing against them, and we corrected a legacy hip-damping law. Say this if it comes up.

## S13 · How one training sample is made

Five steps, all real numbers.

Fit the conditional prior. Roll the condition dice — mass, speed, ground stiffness. Expand the network's dimensionless vector into a physical design using that block's mass. Drop it in the simulator, which returns 16 metrics. Keep the designs that are feasible and on the Pareto frontier — that pair, condition plus design, is one training sample.

One detail worth noting: the acceptance requirements — the g-limit and the stroke limit — are drawn *after* the simulation, not before. So changing the requirement costs us nothing; we re-judge existing drops instead of re-running them.

## S14 · Structural sizing and the manufacturability criterion

The simulator gives peak joint moments and foot axial force. Each segment is then sized as a thin-walled tube with wall thickness one tenth of the diameter. Two numbers come out: **how thick it needs to be**, from material strength; and **how thick it is allowed to be**, from segment length — we cap diameter at a quarter of the segment. If the first exceeds the second, the design is declared unmanufacturable.

We default to carbon-fibre nylon because we want to 3D print. But printed yield strength is only 70 MPa, and after a safety factor of 2 we are left with 35 MPa allowable. Swap the same leg to aluminium 7075 and the tube goes from 22.9 down to 11.6 millimetres, with 39 percent less mass — and every slenderness violation disappears.

**[Q]** I am not confident about that 70 MPa. It comes from printed-polymer data, not from an aerospace material spec. **Is a safety factor of 2 on printed yield the right discipline here, or am I being conservative in the wrong place?**

**[Q — for Prof. Bigoni]** Our manufacturability criterion is strength plus a slenderness cap. **We do not check local buckling of the thin-walled tube.** At t = 0.1D under impulsive axial compression, should we expect local wall buckling to govern before material yield — and if so, what is the right check to add?

## S15 · Ground contact

Ground stiffness enters as a contact stiffness k_c, from the modulus of subgrade reaction times foot contact area, spanning wet sand to hard runway.

One point on the upper bound: the foot is not rigid. A compliant foot pad sits **in series** with the ground, so effective stiffness is capped by whichever is softer. Two mega-newtons per metre transmitted through a 20-millimetre foot pad is not physical, so we pulled the ceiling down to one.

## S16 · Does the model actually learn?

This is the check that the loop is real and not just resampling. We freeze an exam set of conditions the model never trained on, and compare the model's design against a reference Pareto frontier obtained by brute-force sampling in the same condition.

Crossing zero means the model beats the frontier that generated its own training data. It crosses at round 12 and settles around 8 percent better.

## S17 · Mass extrapolation

Hard ground and grass work up to 40 kg; wet sand to 27.8. At the 30 kg anchor all three terrains pass. Above 65 kg everything fails.

⚠ **STALE and important** — the model on this slide was trained on 1 to 12 kg, so everything above 12 kg here is **extrapolation**. We have since retrained on 4 to 36 kg, and the failures at 20 and 30 kg on hard ground disappear. The honest statement today is: *in the retrained model, 5 to 30 kg is covered by training, not by extrapolation.*

This is also the slide where the target shifted. Nobody builds a 1 kg fixed-wing aircraft, and airliner-scale is far outside anything this pipeline reaches. **5 to 30 kg — small fixed-wing UAVs — is where this belongs.**

## S19 · The ledger

I want to show this one deliberately. Every constant in the model, colour-coded by where it came from: blue is measured literature, green is engineering handbook, **orange is a number we chose ourselves**.

The orange rows are where I would like you to push.

Two I already distrust. First, the tube diameter cap at a quarter of segment length — that single number decides which designs are called unmanufacturable, and it is ours, not a handbook's. Second, the touchdown posture: we froze the knee at 90 degrees, but the median in our own annotated video is 133.

⚠ **STALE** — posture is no longer frozen. We made it a design variable, and it turned out to be the single largest lever we have measured: mismatching it changes peak deceleration by a factor of two. The model now converges to about 127 and 139 degrees, and holds those almost independently of condition.

Also on this slide: tau is a Kelvin–Voigt relaxation time in seconds, not a dimensionless damping ratio. The conversion depends on the design, so it must be back-computed case by case. We had this labelled wrongly for a while.

## S22–S24 · Validity of the prior, and real versus generated bones

These compare generated designs against real birds — the same allometric band, and segment-by-segment.

The point is not that they match. It is that we can *say* how far off they are, because the prior is explicit rather than buried in a loss function.

## S25 · Architecture of the cVAE

A small conditional VAE — encoder, three-dimensional latent, decoder — about ten thousand parameters. Deliberately small: the constraint work is done by the prior, not by the network.

---

## Closing · What I want to ask

**1 · Where does bio-inspiration stop?**
We copied geometry only. Of our design variables, the biological prior covers three — L₁, r₂, r₃. The four that actually determine performance — three joint stiffnesses and the damping time — have **no biological content at all**. They are hard to measure in a live bird.

**[Q]** In bio-inspired design generally, is freezing a hard-to-measure biological control variable into a constant standard practice, or is it a fundamental weakness?

**2 · The hard–soft ratio.** (for Prof. Hsia)
Our leg is rigid links plus torsional springs — the hard-to-soft ratio is effectively infinite. A real bird's first stage of cushioning is the foot pad and the soft tissue of the tarsometatarsus, and that layer is simply missing from our model.

**[Q]** Does landing impact have its own characteristic hard–soft band, the way you defined it across species? Should pad stiffness and thickness become an eighth design variable? I did a rough calculation using LCE as a foot pad, and it does seem to improve hard-ground landings.

**3 · Timescales, if we go semi-active.**
Impact is milliseconds; LCE actuation is seconds — three orders of magnitude apart. So LCE cannot absorb the impact itself. And its actuation stress, 0.1 to 1 MPa, is far from the roughly 700 N peak axial force in one leg.

**[Q]** That points to LCE as a *regulator* — setting the leg up before touchdown — rather than a load-bearing element. Is that a useful role at this scale?

**4 · Stability.** (for Prof. Bigoni)
Two places where I suspect we are under-analysed. One is the local buckling question on S14. The other: we recently tested a four-bar linkage that couples two joints, and found the mechanism approaches a singular configuration if the coupler is mounted close to the joint — small input rotations demanding very large output rotations.

**[Q]** We treated that as a geometry problem to avoid. Should we instead be treating proximity to singularity as a *design constraint with a margin*, the way one would treat a buckling load?

---

## Appendix — what changed since this deck was made (2026-08-27)

If you would rather present current results, these are the five things not yet on any slide:

1. **Touchdown posture became a design variable** (9-dim, not 7-dim). It is the largest single lever measured so far.
2. **Hip damping law corrected** — the legacy special case was over-damping the hip and inflating peak loads; the old results were conservative, not optimistic.
3. **Mass range retargeted to 4–36 kg** for a 5–30 kg product range. This removes the extrapolation caveat on S17.
4. **Mechanical architecture ablation** — removing the knee shock absorber, and locking a degree of freedom with a coupling rod, both cost nothing measurable. Architecture can be chosen on manufacturability alone.
5. **A 5 kg prototype specification** — geometry, spring and damper recipes, mounting lever arms, and drop-rig requirements.
