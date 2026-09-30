# -*- coding: utf-8 -*-
"""English script, 10 slides (v3: generation-first). Shared by both decks."""

SCRIPT = [
# 1 Title
"""[~40 s]

Good morning. I am Zihan Wang, working with Professor Zhao and Professor Hsia.

The question is simple: how long should the leg of a landing gear be?
There is no engineering rule for it. We built a generator（生成器）that answers it for any
operating point, and we used birds to tell the generator where to look.

I will show what the generator produces, and then what it learned that we never told it.""",

# 2 Background
"""[~60 s]

First, what we are up against.

Multirotor and VTOL drones in the five to thirty kilogram range almost all carry the same
landing gear: a skid frame（滑橇式支架）with two rubber pads underneath. Simple, cheap, easy
to maintain - and flying in very large numbers in agriculture, inspection and delivery.

But it is nothing like the gear on a large aircraft. Large aircraft have an oleo-pneumatic
shock strut（油气式缓冲支柱）, a whole mature technology. At our scale the gear contains no
separate shock absorber at all. The landing energy can only go into the frame bending a
little and those two rubber pads squashing a little.

That was good enough when payloads were light and the ground was flat. It is not any more:
payloads keep growing, and landing sites are moving from paved pads to turf, wet sand and
uneven ground.

So three problems surface. First, the stroke（行程）is limited - a rubber pad can only be
squashed so far. Second, the landing deceleration is high; the shock reaches the airframe and
the payload almost undamped. Third, and this is the one that matters most to us: there is no
design method for a given operating point. How long the leg should be, how many links, how
stiff the joints - at this scale there is no rule to follow.

Our entry point is the third one. And the first two happen to be exactly what a bird's leg is
good at.

提示：三个问题按图从左到右指一遍，最后一句停在"腿该多长"上，接下一页。""",

# 2 Problem and approach
"""[~90 s]

What we are building.

We design landing gear for vertical take-off drones between five and thirty kilograms. It is
hard for three reasons: the design space is large; every candidate needs a full multibody drop
simulation（多体落震仿真）, so evaluation is expensive; and there is no rule for leg length.

So the target is not one design. It is a machine that takes an operating point（工况）— mass,
landing speed, ground stiffness（地面刚度）, and the acceptance limits — and returns a leg.

Four blocks, left to right.
The data engine turns public data into numbers: bird video gives touchdown posture; a public
anatomy dataset gives a mass-conditioned prior（质量条件先验）for leg length; speed literature
and geotechnical data give the landing conditions; material constants say what can be built.
The physics simulator drops every candidate. Not scored — dropped.
The self-improving loop（自循环）feeds every result back into the training set.
The final system: five numbers in, nine numbers out.

Today: where that prior comes from, what the generator produces, and where it disagrees with
biology.

提示：先讲目标，再顺着四块走，不展开细节。""",

# 3 Generator
"""[~90 s]

Start with what is being designed, so the numbers later make sense.

A three-segment leg. The lowest segment we call $L_1$; the two above it are given as ratios
$r_2$ and $r_3$ of $L_1$. Two touchdown angles, theta-A at the ankle and theta-K at the knee.
Two shock absorbers（减震器）, at ankle and hip; the knee is a free hinge（自由铰）. Each
shock unit has a stiffness（刚度）and the pair shares one relaxation time（松弛时间）.

That is nine numbers. They are the output.

The input is the operating point（工况）: mass, sink rate（下沉速度）, ground stiffness
（地面刚度）, and two acceptance limits — ten g and twenty-four millimetres of stroke.

The generator maps five numbers to nine. It is trained on drop simulations, and every design
it returns is dropped again in the simulator before we accept it.

Of the nine numbers, eight are bounded by mechanics. The one we could not bound is $L_1$.
That is where the birds come in.

提示：先把 9 个输出量指一遍，最后一句留钩子到生物。
被问到"为什么是这个构型"再答：三种架构峰值 3.60 / 3.52 / 3.49 g，落在随机种子噪声内，
性能分不出来，取零件最少的一种。""",

# 3 Biological prior
"""[~90 s]

Where the prior comes from.

We joined three public datasets — morphology, ecology, and the bird family tree — into one
table of eight thousand seven hundred species. Nothing new was measured.

This plot is the waterbirds（水鸟）, two hundred and one species. Both axes are logarithmic
（对数坐标）, so the straight line is a power law（幂律）. The formula is at the top:
leg length grows with mass to the power 0.366.

Why waterbirds: they land the way our drone lands — slowly, on soft and uneven ground — and
they are the birds our perception pipeline（感知管线）was built on.

The band is plus and minus two and a half standard deviations（标准差）. That band is the
prior: for a target mass it gives a range of plausible leg lengths, not a number.

The grey lines are theory — geometric similarity（几何相似）at one third, elastic similarity
（弹性相似）at one quarter. Birds are steeper than both.

One honest point: above twelve kilograms there is no waterbird. The top half of our range is
extrapolation（外推）, which is why the heavy end must be settled by simulation.

提示：公式念一遍；把"走廊不是一个数"说清楚。""",

# 4 Trade-off
"""[~90 s]

The law has structure. Same mass, different leg — and it is not random.

There is a number called the hand-wing index（手翼指数）, HWI: the shape of the wing tip.
High HWI means a pointed wing, which means cheap, efficient flight.
The formula is at the top. Using body mass alone to explain leg length gives R squared
（决定系数）of 0.671; adding the HWI term takes it to 0.783 - one term, across all 8,870
species. The sign is negative: for the same mass, the better flier has the shorter leg.
To put a number on it: going from a sparrow's HWI of 17 to a waterbird's 44, twenty-seven
units, corresponds to a leg about thirty percent shorter.
This is ordinary least squares, with no phylogenetic control（亲缘校正）yet - that is the
next slide.

The clearest case: two falcons（隼）, same family, both raptors, both fly.
The sooty falcon hunts on the wing in open sky. The leg is dead weight（死重）. Thirty-four
millimetres.
The forest-falcon flies in short bursts inside the canopy and must perch and grip.
Sixty-two millimetres — while being sixty-five grams lighter.

Across all sixty-two falcon species the correlation is minus 0.68.

Read it as a trade: the leg is mass carried in flight. Invest in flight, pay with leg.
That is the same trade a landing-gear designer makes: stroke（行程）against carried mass.

提示：两根骨头指给他们看，然后一句话接到工程权衡。""",

# 5 Robustness
"""[~75 s]

A biologist will object: closely related species are not independent samples（独立样本）.
We checked three ways.

First, phylogenetic regression（系统发育回归）on two hundred trees. Shared ancestry explains
sixty-five percent of the naive effect. The remaining thirty-five percent survives on every
tree — p below ten to the minus fifty-six.

Second, inside families（科）: remove mass, split each family into better and worse fliers,
compare legs. Seventy-seven of ninety-six families go the predicted way.

Third, across the tree: twenty-four independent clades（支系）that changed leg length on their
own stem. Ten of the twelve better-flying clades shortened; six of the twelve poorer ones.

So the trade-off is not one ancient accident. It is repeated.
That is what makes it usable as a prior.

提示：三个数，一页过，不展开方法。""",

# 7 Generated designs
"""[~120 s]

What it produces.

Three operating points across the envelope（包线）: five kilograms on turf, twelve on rigid
ground, thirty on wet sand. Same sink rate.

For each, the generator returns a leg. Drawn to a common scale. Under each leg, the nine
numbers it chose, and the drop-test result.

Five kilograms: ninety-seven millimetre foot bone, 3.8 g peak, 47 grams of leg.
Twelve kilograms on concrete: 125 millimetres, 4.1 g, 101 grams.
Thirty kilograms on sand: 185 millimetres, 3.8 g, 383 grams.
All three pass — under ten g, under twenty-four millimetres.

These are not the best designs. Each is the median of the feasible set（可行集）, so this is
typical output, not a cherry-picked one.

Across the whole envelope — nineteen masses, six terrains, two random seeds — eighty-three
to eighty-seven percent of first-shot designs pass the drop test.

And one comparison with biology: at the mass and conditions of seven real waterbirds, the
generated leg passes ninety-five percent of the time; the real bird's own leg, seventy-one.

提示：这是全场核心页，讲慢。三条腿一条一条念数字。""",

# 8 Learned without being told
"""[~120 s]

What it learned that we never told it.

First, plainly: the power law is in our parameterisation（参数化）, not a discovery. The
search box is written in the bird's coordinates, and the generator chooses an offset u from
the bird's line. If it had no opinion about mass, u would be flat.

It is not flat. The effective exponent（有效指数）depends on the ground. On soft ground it
stays near the bird — 0.34 against 0.39. On rigid ground it drops to 0.26. No design sits
near the corridor wall; the generator stops on its own.
The reason is physical: on concrete the ground returns nothing, so the machine buys shock
absorption with joint stiffness instead of leg length. A bird, with bone and tendon（骨头和
肌腱）, cannot make that trade.

Second, the right plot. Segment ratios（分段比）and posture were given as the full measured
range, centre not privileged. On the two segment ratios the generator lands on the bird's
median to within one percent of the box — on three terrains and two seeds. Nothing in the
physics knew the bird's value.
On posture it disagrees: it crouches seventeen degrees lower at the ankle. A bird's ankle
also has to walk and swim.

Third — not on this slide — it found the edge of the envelope. At two metres per second on
rigid ground, no design passes both limits. An energy bound（能量下界）shows why: stopping
two metres per second in twenty-four millimetres needs 8.5 g on average, and any real force
profile peaks above ten. We moved that limit to 1.5 metres per second.

提示：先承认幂律是给的，再讲分段比这个真正独立的证据。""",

# 9 Real swan vs generator
"""[~85 s]

So far this has all been laws and statistics. Now something more direct: test it against a
real bird.

The trumpeter swan（号手天鹅）, 11.07 kilograms - the largest waterbird in our database.
Its leg geometry is entirely measured: the tarsus（跗跖骨）length of 108.5 millimetres comes
from AVONET specimen measurements, the segment ratios come from Watanabe's 2017 skeletal work
on ducks and geese, and the touchdown angles come from our own twelve waterbird landing
videos. Nothing here is extrapolated（外推）. This bird is real.

Then I asked the generator for a leg at the same body mass and the same operating point.

One thing must be said first: the joint stiffness（关节刚度）is identical on both sides.
There is no measured stiffness data for a real bird's leg anywhere - none. So I gave the
swan's leg exactly the stiffness the generator chose. The only variables on this slide are
geometry and touchdown posture（触地姿态）.

The result. On the left, the swan: peak 7.49 g. On the right, the generator: 4.19 g,
forty-four percent lower. Leg mass drops from 127 grams to 93, twenty-seven percent lighter.

It did two things. It made the leg twelve percent longer, 108 to 122 millimetres. And it
lowered the ankle angle（踝角）from 144 degrees to 128 - it crouches more.
Those two changes buy a forty-four percent reduction in peak deceleration.

This also answers the seventeen degrees from the previous slide: that is what those degrees
are worth in physics.

The honest note: a swan's leg did not evolve for concrete. It lands on water and mud. So this
is not "we beat the swan". It is that when you change the ground, the best posture is no
longer the bird's - and the generator finds the new one by itself.

提示：先把"刚度是共用的"说清楚，再念两个数，最后一句留给"换了地面最优就变了"。""",

# 10 与 DJI Agras T30 式起落架对比
"""[~80 s]

Finally, a comparison with what engineers use today.

First, the boundary of the comparison. Our physics simulation is still working on horizontal
sliding（水平滑动距离）, so for now we only compare vertical landing. If the drone only comes
straight down, it does not need a wheel that rolls. Both sides are pure vertical drops, so the
comparison is fair.

The reference is the DJI Agras T30, one of the most widely used agricultural drones. It weighs
26.4 kilograms empty, which matches our 30 kilogram case. From its published dimensions I
rebuilt its landing gear: a skid frame with rubber foot pads（滑橇加橡胶脚垫）. Two arches -
four splayed struts（斜撑）and two skid tubes（滑橇管）- with four rubber pads underneath.
The frame uses exactly the same printed material as our leg and the same safety factor of two,
sized against axial stress and buckling（屈曲）. For the pads I took forty millimetres diameter
and twenty millimetres thick, a normal size for this kind of pad. DJI does not publish it, so
that number is my estimate, and it is marked on the slide.

Three bars on the left. The green band is the acceptance range（合格区）.
Mass: 709 grams against 669 grams - essentially the same, and there is no limit on mass here.
Peak deceleration: 33 g against 7.9 g - the skid is outside, we are inside.
Available stroke: 4.7 millimetres against 22.6 - the same picture.
The difference is not weight. It is deceleration.

The right plot gives the reason, and the reason does not depend on the configuration. First,
what the horizontal axis means: available stroke（可用行程）is how many millimetres the gear
itself is compressed vertically after the foot touches down - for our leg, the three links
folding; for the skid, the rubber pads being squashed. All of the kinetic energy has to be
absorbed over that distance, so the shorter it is, the larger the force.

Our acceptance limits are peak below ten g and stroke below twenty-four millimetres. Turning
that around, ten g needs at least eighteen millimetres. So the pass window is eighteen to
twenty-four millimetres - narrow. A0 sits at 22.6, inside the window. The skid has about five
millimetres, and no pad size reaches more than seven - an order of magnitude short.

A skid locks stiffness and stroke to the size of the pad. Articulation plus damping separates
them.

提示：先说"只比垂直着陆"这个边界，再念三根柱子，最后用右图解释为什么。""",

# 11 Summary
"""[~45 s]

Three things to take away.

Birds follow a tight leg-length law, and inside it, flight efficiency trades against leg
length. That trade-off survives phylogenetic control.

We hand that law to a generator as a corridor, not a number. The generator returns verified
designs across the envelope, at eighty-plus percent first-shot feasibility.

And where the generator departs from biology — on rigid ground, and on posture — it departs
for reasons we can name.

Open: a rebound criterion（回弹判据）in the acceptance test, horizontal touchdown speed as an
input, and hardware.

Thank you.""",
]
