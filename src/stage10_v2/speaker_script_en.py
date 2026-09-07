# -*- coding: utf-8 -*-
"""英文演讲稿。中英两版 PPT 共用这一份 —— 改这里，两个 PPT 重跑即可同步。

结构（13 页 + 落震视频页可选）：
  1 问题 · 2 数据 · 3 定律一 · 4 先验 · 5 定律二 · 6 PGLS · 7 隼科
  8 科内 · 9 跨支系 · 10 功能检验 · 11 交给机器 · 12 A0 · 13 机器自己走出这条律
  (14) 一次落震：这笔交易值多少
配速：每段开头 [~xx s]，13 页约 14 分钟，14 页约 15.5 分钟。
"""

SCRIPT = [
# ---------------- 1 首页 ----------------
"""[~40 s]

Good morning. My name is Zihan Wang. I work with Professor Zhao and Professor Hsia.

Today I want to answer one question. How long should the leg of a landing gear be?
A landing gear is the leg a drone stands on when it lands.

We do not have a good rule for this. So we asked the birds.
Birds have solved this problem many times, over a very long time.

The talk has two halves.
First: birds follow a law, and we tried hard to break it.
Second: we gave that law to a machine, and watched what the machine did with it.

提示：第一页只把问题和两半结构说清楚。""",

# ---------------- 2 数据集 ----------------
"""[~60 s]

First, the data. We did not measure any new bird. We joined three published datasets
（把三个已发表的公开数据集拼成一张表）.

The first one gives shape: leg length, wing length, body mass.
The second one gives behaviour: what the bird eats, and where it looks for food.
The third one is the family tree of all birds（鸟类的系统发育树，记录谁和谁亲缘更近）.

The names in these three datasets are not the same. So we had to match them one by one.
About ninety-nine percent matched.

The result is one table. Eight thousand seven hundred species.
Every row has shape, behaviour, and a position on the tree.

This table is the base of everything after this slide.

提示：强调"我们没有测新数据，价值在这次拼接"。""",

# ---------------- 3 定律一 ----------------
"""[~75 s]

Now the first law. Heavier birds have longer legs. That is not a surprise.

On this plot both axes are logarithmic（对数坐标）. So a straight line means a power law（幂律）.
The slope of that line is called b. For waterbirds（水鸟）, b is 0.366.

Look at the green line and the green band.
The band is one standard deviation（一个标准差）wide. It is thin. The law is tight.

But the interesting part is not the line. It is the distance from the line.

On the right, I subtract the line from every bird. What is left is called the residual
（残差，就是"实测减掉公式预测"剩下的部分）. I call it u.

If u is below zero, this bird has a shorter leg than its body mass predicts.
If u is above zero, its leg is longer.

Everything after this slide is about u.

提示：这一页只要让听众记住 u 是什么。""",

# ---------------- 4 先验 ----------------
"""[~60 s]

Here is the same law in normal units, so you can read millimetres directly.
The formula is at the top. This is the number we actually use.

The red line is the waterbird fit. Around it is a band of plus and minus two point five
standard deviations.

Why waterbirds? Because they land the way our drone lands: slowly, on soft and uneven
ground. They are also the birds our camera pipeline（感知管线）was built on.

The two grey lines are theory. One is geometric similarity（几何相似）, slope one third.
The other is elastic similarity（弹性相似）, slope one quarter. Birds are steeper than both.

Green is our product range: five to thirty kilograms.
And an honest point: above twelve kilograms there is no waterbird. Swans and pelicans stop
there. So the top half of our range is extrapolation（外推）.

提示：公式念一遍，然后把水鸟和感知管线的关系说一句。""",

# ---------------- 5 定律二 ----------------
"""[~75 s]

Now the second law. This one is new.

There is a number called the hand-wing index（手翼指数）, or HWI. It describes the shape of
the wing tip. A high HWI means a pointed wing. A pointed wing means cheap, efficient,
long-distance flight.

On the left, I predict leg length from body mass only. R squared（决定系数，衡量公式解释了
多少）is 0.67.

On the right, I add HWI. R squared goes to 0.79.
One extra term removes more than a third of the error that was left.

The coefficient is negative. So, for the same body mass, a bird that flies better has a
shorter leg.

Think of it as a trade. The leg is weight you carry in the air.
If you invest in flight, you pay for it with leg.

This is exactly the trade a landing gear designer makes: stroke（行程）against carried mass.

提示：这里第一次把"生物权衡 = 工程权衡"说出来，后面要回收。""",

# ---------------- 6 PGLS ----------------
"""[~90 s]

Now the objection. A biologist will say: your birds are not independent（不是独立样本）.

Two swifts look the same because they share an ancestor（共同祖先）, not because each one
solved the problem by itself. If I count them as two separate points, I am cheating.

The standard fix is called PGLS（系统发育广义最小二乘）. It puts the family tree into the
error term of the regression.

The tree is not known exactly, so we did not use one tree. We used two hundred trees, from
two different backbones（两套骨架，两种建树方案）.

On the left: the naive number is minus 0.078. After the correction it is minus 0.027.
Shared ancestry explains about sixty-five percent of what we first saw.
That is a big cut. We report the small number, not the nice-looking one.

In the middle: but it never reaches zero. Not on a single tree out of two hundred.

On the right: lambda（λ，衡量亲缘对这个性状的影响有多强）is 0.93, almost one.
This trait is almost completely controlled by ancestry. That is why the correction had to
be so strong — and why the part that survives it matters.

提示：力学背景的听众记住"缩水 65%，但没归零"就够。""",

# ---------------- 7 隼科 ----------------
"""[~90 s]

That was statistics. Now one concrete case. This is the easiest slide today.

Both of these birds are falcons（隼）. Same family. Almost the same ancestors.
Both eat meat. Both fly.

On the left, the Sooty Falcon. It migrates long distances and hunts in open sky, on the
wing. For this bird the leg is dead weight and drag（死重和阻力）.
Its leg bone is thirty-four millimetres.

On the right, the Forest-falcon. It lives inside the Amazon forest. It flies in short
bursts between branches, and it must stand and grip.
Its leg bone is sixty-two millimetres.

Here is the point. The forest-falcon is sixty-five grams lighter.
Lighter body, and eighty-one percent more leg.

So the difference cannot come from body size, and it cannot come from ancestry.
It can only come from how they fly.

On the right of the slide are all sixty-two falcon species. Same trend.
The correlation is minus 0.68.

提示：全场最好懂的一页，讲慢，把两根骨头指给他们看。""",

# ---------------- 8 科内 ----------------
"""[~60 s]

Now I repeat the falcon test ninety-six times.

For each bird family（科）, I first remove the effect of body mass.
Then I cut the family into two halves: the better fliers, and the rest.
Then I compare their legs.

Each bar is one family. Blue means the better fliers have shorter legs.

Seventy-seven families out of ninety-six go this way. By chance it would be about forty-eight.

And in the green box: if I make it stricter and stay inside a single genus（属，比科更近一级
的分类）, it still holds. Two hundred fifty-one out of four hundred nineteen.

提示：一句话 —— "隼科那件事，重复了 96 遍"。""",

# ---------------- 9 跨支系 ----------------
"""[~75 s]

Same idea, different scale. Now the whole tree.

Each bar is a clade（支系，树上的一个分支）that changed its leg length on its own branch.
The grey line shows how well that clade flies.

One warning first. At the beginning I found forty such changes. But sixteen of them were
inside another one — the same event counted twice. So I kept only the twenty-four that do
not contain each other. After removing the double counting, the result became stronger,
not weaker.

Left half, the good fliers: ten out of twelve made the leg shorter.
Right half, the poor fliers: only six out of twelve.

The exceptions are worth naming. Grebes and pelicans went the other way. They dive and
they wade（潜水和涉水）. For them the leg has a second job in the water.
That is the next slide.

提示：主动说自己纠正过计数，这一条反而加分。""",

# ---------------- 10 功能检验 ----------------
"""[~60 s]

So let me check the function directly, with a completely separate dataset.

This dataset records, for every bird, where it looks for food: in the air, in water, in
trees, or on the ground.

On the left: birds that feed in the air have the shortest legs, by far.
Birds that walk on the ground have the longest.

On the right, I hold body mass fixed and measure the correlation.
Ground feeding is positive. Tree and air feeding are negative.
The sign changes exactly where function says it should.

Water is the weak one, only minus 0.06. That is honest: water birds are not one group.
A wading heron and a diving cormorant are both "water", and they need very different legs.

提示：把最弱的那条自己讲出来，比被问出来好。这里结束生物部分。""",

# ---------------- 11 交给机器 ----------------
"""[~60 s]

That is the biology. Now, what do we do with it?

This is important: we are not copying a bird.

The law gives a corridor（走廊，一段合理范围）. For a target mass it says which leg lengths
are plausible. It does not say which one to build.

Inside that corridor, a generative model（生成模型）proposes designs. It takes five numbers
describing the situation — mass, landing speed, ground stiffness（地面刚度）, and the
acceptance limits — and returns nine numbers describing a leg.

Then every candidate is tested by a full multibody simulation（多体动力学仿真）.
Not by a score. By a real drop test inside the computer.

Biology narrows the search. Physics makes the decision.

提示：力学背景的听众最在意这一页，"不是抄鸟"要说清楚。""",

# ---------------- 12 A0 ----------------
"""[~55 s]

One page of engineering. This is the leg we simulate.

Three segments. We name them after the bird: foot bone, shin bone, thigh bone.

Two shock absorbers（减震器）: one at the ankle, one at the hip.
The knee is a free hinge（自由铰）— no spring, no damper（没有弹簧，也没有阻尼器）.

Why only two? We tested three architectures（架构）. Peak deceleration（峰值过载）was
3.60, 3.52 and 3.49 g. But if I only change the random seed（随机种子）, I already get three
percent. So the differences are inside the noise（噪声）.

If performance cannot choose, manufacturing chooses. This one has the fewest parts, and it
gives a thirty-six millimetre lever arm（力臂）at the thigh — that turns a special-order
spring into a catalogue part.

提示：结论是"性能分不出来，所以按制造性选"。""",

# ---------------- 13 机器自己走出这条律（核心） ----------------
"""[~110 s]

This is the result I care about most.

Remember what the generator knows. For each body mass, it searches its own design space and
returns a leg. It is never told that leg length should follow a power law of mass.
It solves each mass separately.

Now look at the left plot. I take every design that passed the drop test, take the median at
each mass, and plot it. Both axes are logarithmic again.

The points fall on a straight line. R squared is 0.98 to 0.99.
So the machine did not just find good legs. It found a law — the same kind of law we found
in the birds. Nobody put that law into it.

Now the exponent, on the right.
Elastic similarity is 0.25. Geometric similarity is 0.33. Real waterbirds are 0.366.

On soft ground — wet sand and turf, which is the ground waterbirds actually land on — the
machine chooses 0.34 and 0.35. That is almost the bird's number, and it is reproducible
across two random seeds.

On rigid ground, concrete, it drops to 0.263. That is close to elastic similarity.

So: same form of law, and on the bird's own terrain, almost the same number.
Where it departs, there is a physical reason. On concrete the ground gives nothing back. The
machine then buys the shock absorption with joint stiffness and damping（关节刚度和阻尼）
instead of buying it with leg length. A bird cannot do that. A bird has only bone and tendon
（骨头和肌腱）. So the bird needs the extra length. Our machine does not.

That is the sentence I want you to take away: the machine agrees with biology on soft ground,
and it tells us exactly where biology stops being the right answer.

提示：这是全场落点。左图说"形式一致"，右图说"软地上数值也一致，硬地才分开"。""",

# ---------------- 14 一次落震（有视频时才用） ----------------
"""[~75 s]

Finally, what does that difference cost, in one landing?

Both legs carry the same thirty kilogram body, at the same landing speed, on rigid ground.
Same joints — same springs, same dampers. Only the exponent is different.
Green uses the bird's 0.366. Blue uses the machine's 0.263.

Watch the numbers at the bottom.
The peak deceleration is almost the same. The stroke is almost the same.
The difference is leg mass.

The machine pays a little in shock and gets a lighter leg. Both stay inside our limits of
ten g and twenty-four millimetres.

Three things are still open. A rebound criterion（回弹判据）in the acceptance test.
The horizontal landing speed, which is not yet an input. And hardware.

Thank you.

提示：视频放完再说数字。最后一句停一下再说 Thank you。""",
]
