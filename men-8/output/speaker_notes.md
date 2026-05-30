# Speaker Notes: Slides 15-20

## Slide 15: Predictive Performance Table

> "Before we can estimate how price affects demand, we first need to check: can our AI models actually predict sales rank and price well?"

- Both tables show how accurate different models are at predicting sales rank (Q) and price (P). Higher numbers are better.
- The first couple rows are the same on both sides because those are basic models that don't use any embeddings yet, just raw numbers like ratings and review counts.
- The row highlighted in green is the important one. That's when we add our AI embeddings to the model. Accuracy jumps from around 0.52 to 0.78 for predicting sales rank. Huge improvement.
- Why does this matter? Because DoubleML works in two stages. The first stage needs to predict sales rank and price accurately. If it can't, the final elasticity number won't be reliable. These high scores tell us: yes, our embeddings are doing their job.
- Both women's and men's models show the same pattern. Embeddings help a lot, especially for price prediction.

---

## Slide 16: Elasticity Forest Plots (Women vs Men)

> "Now here's the main result. These forest plots show our answer to the core question: when price goes up, what happens to demand?"

- Each dot is an elasticity estimate from a different model setup. The horizontal lines are confidence intervals, basically the range where the true value probably sits.
- Left side is women's shoes, right side is men's.
- For women's, the estimates land around **-0.06 to -0.09**. For men's, they're a bit bigger at **-0.07 to -0.10**.
- What do these numbers actually mean? Take the men's number of -0.095. That means: if a shoe's price goes up by 1%, its sales rank gets about 0.095% worse. It's a small effect, but it's real and statistically significant.
- None of the confidence intervals cross zero. That's the important part. Zero would mean "price has no effect on demand." All our estimates are clearly negative, confirming that higher prices do hurt sales.
- The fact that we get similar results for both genders, using the exact same method, gives us confidence that this isn't a fluke.

---

## Slide 17: Elasticity Slope Scatter Plots

> "These scatter plots show what that elasticity looks like in the actual data."

- Every dot is one product at one point in time. The x-axis is price (in log scale), y-axis is sales rank (also log scale).
- The orange line is the DoubleML elasticity slope. The shaded area is the confidence band.
- The line tilts slightly downward, meaning higher price goes with worse sales rank. That's the price elasticity we just talked about.
- You might notice the slope looks almost flat. That's actually the point. Remember the EDA slide earlier where the raw OLS slope was about -0.6? That was heavily biased because it didn't control for product quality. After DoubleML removes all the confounding, the true causal effect is much smaller: -0.066 for women, -0.095 for men.
- In everyday terms: raising your price a little bit doesn't tank your sales. Demand is **inelastic** for shoes on Amazon, which is typical for branded consumer goods.

---

## Slide 18: CATE for Women's Shoes

> "So far we've been looking at the average effect across all shoes. But does every shoe react the same way to price changes? This slide says: not really."

- CATE just means "how price-sensitive is each individual product." Instead of one average number, we calculated a separate number for every product.
- Each panel is one cluster of similar shoes. The y-axis shows how elastic that product is (more negative = more price-sensitive), and x-axis is the price.
- For women's shoes, the clusters are actually pretty similar. Most hover around -0.13 to -0.16. There's some variation, but nothing dramatic.
- The interesting pattern within each cluster: more expensive products tend to be more price-sensitive (dots trend downward to the right). Makes sense, shoppers pay more attention to price when spending more money.

---

## Slide 19: CATE for Men's Shoes

> "Now here's where it gets really interesting. Men's shoes tell a very different story."

- Same analysis as the previous slide, but for men's shoes. And look at how much more spread out the clusters are.
- **Cluster 0**, the budget walking shoes around $31. CATE is only -0.12. These buyers don't really care about small price changes. They're buying cheap shoes and not overthinking it.
- **Cluster 1**, loafers and dress shoes around $42. A bit more sensitive at -0.15, but still pretty mild.
- Now look at **Cluster 2**, premium athletic shoes around $87. CATE jumps to **-0.34**. These shoppers are comparing options. If you raise the price, they notice and they leave.
- **Cluster 4** is even more extreme at **-0.37** for premium performance shoes around $94.
- The orange box at the bottom sums it up: **premium shoes are 3 times more price-sensitive than budget shoes**. That's a huge difference.
- For pricing decisions, this means: you can be more aggressive with pricing on a $30 walking shoe. But if you're pricing a $90 running shoe, even a small increase could cost you meaningful sales.

---

## Slide 20: CATE Distributions Side by Side

> "Last comparison slide. Let's put women's and men's next to each other and see the full picture."

- Top left: women's CATE distribution. The average is -0.142, and most products fall in a narrow band between -0.05 and -0.25. Pretty uniform.
- Top right: men's CATE distribution. The average is -0.213, and the spread is much wider, from -0.66 all the way to +0.24. Much more variation.
- What does that tell us? The men's shoe market has more distinct segments with very different price sensitivities. Budget and premium shoes behave completely differently.
- Women's shoes are more homogeneous. Whether it's pumps, flats, or sneakers, the price sensitivity doesn't vary as dramatically.
- Bottom row shows the same story through a different lens: the sorted effects by cluster. Women's bands are tight, men's bands are wide.
- The practical takeaway for Amazon's pricing team: a one-size-fits-all pricing rule might work OK for women's shoes, but it would be a mistake for men's shoes. Men's shoes really need segment-specific pricing strategies.
