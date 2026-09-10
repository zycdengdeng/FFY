# Week 7 · English speaker script（终版 11 页）

> 括号里的中文是专业词汇注释，不用念。每页开头的秒数是目标时长，合计约 13.5 分钟。


## p1 · Title

[~30 s]

Hello everyone. I am happy to share this work with you - it is still in progress（还在进行中）.
My name is Zihan Wang. I am a first-year PhD student with Professor Zhao.

The topic is landing gear（起落架）for small drones, and how to design it by learning from birds.


## p2 · Background

[~75 s]

First, the background.

Today, most small drones land vertically（垂直着陆）- both multirotors（多旋翼）and VTOL
fixed-wing aircraft（垂直起降固定翼）. Their landing gear is very simple: a skid frame（滑橇式支架）
with rubber pads（橡胶脚垫）underneath. It is cheap and simple, but it has almost no shock
absorber（缓冲机构）. So the landing speed（着陆速度）has to stay low.

And there are aircraft like the ScanEagle. It is a fixed-wing drone, but it cannot really land
by itself. It needs an extra device - a "sky hook"（天钩）- to catch it in the air.

So we asked a question: can a waterbird's leg（水鸟的腿）give us an idea? Could a drone land in a
more graceful way（更优雅地着陆）, without losing flight efficiency（飞行效率）?

This bird-inspired structure（仿生结构）could improve landing for multirotors, and it might give
fixed-wing drones a real way to land.

提示：三张图从左到右各指一下；最后一句停在"能不能让固定翼真正着陆"。


## p3 · What AI can already do (1)

[~60 s]

Before the details, let me show how AI comes into this.

Here is an everyday example. We ask an AI model for a picture, with four requirements（要求）:
a cat, a Siamese cat（暹罗猫）, looking at me, with blue eyes.

The AI gives us a picture that matches all four. And the things we did not say - for example,
that the cat is standing in a home - the AI fills in by itself（自动补齐）. It fills them in from
the data it has learned（学过的数据）.

This is called conditional generation（条件生成）. We give the conditions（条件）; the model
matches them with what it has learned, and returns a solution（解）.

So - if we want a landing gear inspired by birds, for one specific aircraft, what should we do?

提示：把"四个要求 = 条件"、"没说的 AI 自己补 = 先验"这两句说清楚，下一页直接对应。


## p4 · What AI can already do (2)

[~75 s]

Same idea for our problem. We give the model conditions（条件）. And, because we are the
teachers, we also give the model its "textbooks"（书本）- data it can learn from.

What I give the network right now:
First, bird legs cut out of online videos（网络视频）- a perception pipeline（感知管线）finds the
leg joints（关节）in each frame.
Second, biological data（生物数据）from open datasets - how long a bird's leg is, for a bird of a
given weight.
Third, some conditions for the aircraft（飞机的工况）- for example the ground stiffness（地面刚度）
at landing, and the mass of the aircraft（飞机质量）.

With these, the model outputs a bird-inspired design（仿生设计）that fits the conditions. In other
words: this is the design that, based on bird data, the model believes works best for landing
this aircraft.

提示：三个输入对着图上三个竖标签指一遍。


## p5 · Problem and Approach

[~75 s]

This is the pipeline（流水线）.

Let me start at the bottom right: this is what the final system does. You give it operating
conditions（工况）- mass, landing speed, ground type, and the limits on deceleration and stroke.
It returns the parameters（参数）of a bird-inspired leg: three bone lengths, three joint
stiffnesses（关节刚度）, a damping ratio（阻尼比）, and two joint angles.

Going from left to right: the data engine（数据引擎）turns public data into numbers. A physics
simulator（物理仿真器）- Exudyn - drops every candidate design and records what happens. And the
generative model（生成模型）learns from the results.

There is still a lot of room to improve here: which physical quantities to constrain, and how
tight those constraints should be. This part is about mechanics（力学）and mechanical structure
（机械结构）, so it needs collaboration with people who know that field - that is beyond my own
background, and it is where I most need advice.

提示：右下角先讲，再从左往右扫一遍；最后一句把"需要力学指导"说成合作邀请，不是道歉。


## p6 · Generator: Inputs, Outputs, Verification

[~75 s]

This is the mechanical model（机械模型）we simulate. A three-segment leg（三段腿）with an
ankle, a knee and a hip.

On the right: blue is the output（输出）- the nine numbers the model must produce. Green is the
input（输入）- the five numbers that describe the landing. At the bottom are the acceptance rules
（验收条件）: they tell the model which designs are wrong and which are right.

One practical difficulty. A bird's leg has three segments（三段）. But the top segment, L3, and
part of L2, are hidden inside the body. So most data only gives us an accurate number for L1.
For the other two, we can only use ratios（比例）from anatomy datasets（解剖数据集）.

And joint stiffness（关节刚度）and relaxation time（松弛时间）? There is no bird data for these
at all. We give the simulator a range（区间）, it tests many values, and only the ones that pass
the landing test are kept for training（训练）.

提示：把"L3 藏在体内"和"刚度没有鸟的数据"这两个坦白讲清楚，Q&A 会省很多事。


## p7 · A Biological Prior for Leg Length

[~75 s]

This is one rule we put into the model.

We fitted（拟合）the relation between L1 and body mass（体重）, using an open dataset of
waterbirds（水鸟）. Both axes are log scale（对数坐标）, so a straight line here means a
power law（幂律）: leg length grows with mass to the power of 0.37.

Why waterbirds? Two reasons. Our data sources were limited at the start. And waterbirds skim
（滑行）on the water when they land - that is closer to a fixed-wing aircraft.

The line is the fit. The band（带）around it comes from how scattered the data is. This band is
the prior（先验）: for a given mass, it allows the model to search in a range of leg lengths -
not one fixed value - and then the physics decides the best one inside that range.

提示：念一遍公式，指一下 30 kg 处的斜纹区，说"12 kg 以上没有鸟，这是外推"。


## p8 · Flight Efficiency Trades Against Leg Length

[~90 s]

After our last discussion, Professor Hsia pointed out that evolution（进化）also matters for
leg length. So I went back to the bird family tree（鸟类系统发育树）.

I found that a number biologists use for flight efficiency（飞行效率）- the hand-wing index
（手翼指数）, HWI - is related to leg length across the whole tree.

Here is one example. Falco concolor and Micrastur plumbeus come from the same ancestor（祖先）.
Falco concolor hunts by fast, long-distance flight. Micrastur only moves between trees and often
stands on branches. So although Falco concolor is heavier, its leg is shorter - 34 millimetres
against 62.

This pattern holds across the whole tree. Inside one family（科）, the birds that fly better
usually have shorter legs, once you remove the effect of body mass（扣掉体重的影响）.

This opens up the research. Diet（食性）, evolutionary history, way of life - each can be a
parameter, or a factor to remove（干扰项）, so that we get a cleaner geometric law（几何规律）
for a bird's leg.

提示：注意方向：烟隼更重、腿更短；林隼更轻、腿更长。指着两根骨头念数字。


## p9 · Designs from the Generator

[~75 s]

With the dataset and the biological prior, we can train the model. These are some structures
（结构）from our current demo model.

Given specific conditions（工况）, the model outputs a bird-inspired design, and the physics
simulator（物理仿真器）reports how it performs.

Notice that we can even produce a structure for a 30-kilogram "bird" - or aircraft. There is no
30-kilogram bird; the dinosaurs are gone. This means we also thought about extrapolation（外推）
- letting the model reason about data it has never seen. That is one strength of a generative
model（生成模型）. After all, the questions we ask ChatGPT every day were not always asked by
someone before.

But an extrapolated design still needs checking. So on the next page we test the model against a
real bird, at a mass where we do have data.

提示：三条腿从轻到重指一遍；最后一句留钩子到天鹅。


## p10 · A Real Swan Leg vs the Generator's Design

[~80 s]

This is a demo（演示）: a design at 11.1 kilograms, next to the leg of a real 11.1-kilogram
waterbird - a trumpeter swan（号手天鹅）.

One thing first: the joint stiffness（关节刚度）is the same on both sides. There is no bird data
for stiffness, so we gave the swan the same values as the generated leg. Only the geometry
（几何）and the touchdown posture（触地姿态）are different.

Because our model only cares about one thing - landing - its design is better for landing.
Bottom left, peak deceleration（峰值过载）: the blue curve, the generated leg, is much flatter.
Bottom right, leg stroke（行程）: the generated leg uses as much stroke as it can. That means a
softer, more graceful landing（更优雅的着陆）.

The numbers: peak 7.5 g against 4.2 g - 44 percent lower. Leg mass 127 grams against 93 - 27
percent lighter.

提示：先说"刚度共用"，再念两个数；被问"你比天鹅强？"就答"天鹅不是为硬地进化的"。


## p11 · Comparison with DJI Agras T30-Style Gear

[~80 s]

Last, a comparison with today's industry.

Our bird-inspired structure is 3D printed in carbon-fibre-reinforced nylon（碳纤维增强尼龙）.
DJI's agricultural drone, the Agras T30, uses a simple skid（滑橇）with rubber pads. So we asked:
if we build a T30-style landing gear from the same printed material, and simulate（仿真）it in
the same landing, how does it compare?

One boundary: our simulation still only covers vertical landing（垂直着陆）- we are adding
horizontal sliding（水平滑动）now. So this compares vertical landing only.

The result. Total mass is about the same - even slightly in our favour. Peak deceleration（峰值过载）:
33 g for the skid, 7.9 g for ours. Stroke（行程）: 4.7 millimetres against 22.6.

Lower peak, more stroke, a softer landing at the same weight. So this bird-inspired structure can
offer real improvement over current industrial designs.

That is where we are. Thank you.

提示：先说"只比垂直着陆"的边界，再念三根柱子；最后一句就是结束语，不再另有总结页。
