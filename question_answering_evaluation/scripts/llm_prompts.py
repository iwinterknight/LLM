REASONING_CLASSIFICATION_PROMPT = """
Given a set of cooking recipe related question answers we want to categorize questions into commonsense reasoning categories. There are 2 main categories :
1. Physical Commonsense Reasoning
2. Temporal Commonsense Reasoning 

For Physical Commonsense Reasoning there are 2 sub-categories : 
1. Kitchen Tool Physical Commonsense
2. Ingredient Physical Commonsense

For Temporal Commonsense Reasoning there are 3 sub-categories : 
1. Duration Temporal Commonsense
2. Ordering Temporal Commonsense
3. Frequency Temporal Commonsense

Given below (delimited by triple backticks) are various types of commonsense reasoning categories and their sub-categories, with descriptions.

```
1. Physical Commonsense Reasoning Questions :
Description : A physical reasoning question is one which involves an inherent understanding of the physical nature of objects in order to come up with an answer. These could also include questions related to safety in the kitchen. Examples of questions requiring physical reasoning might be:

Kitchen Tool Physical Commonsense Question: "I do not have a scraper or a chopper. How do I scrape the bowl?"
Answer: "You could use any other sharp object like a kitchen knife or even a fork"

Kitchen Tool Physical Commonsense Question: "I have a small vessel, can I boil the water in that?"
Answer: "You will need a somewhat large vessel to prevent the water from spilling over while boiling"

Ingredient Physical Commonsense Question: "I don't have an extra box and the packet of oats did not come in a ziplock bag. How should I store the left-over oats?"
Answer: "You could try tying the cut portion of the packet or sealing the opening with a rubber band"

Kitchen Tool Physical Commonsense Question: "Do I have to put on oven mitts while taking out the turkey from the oven"
Answer: "Yes"

Kitchen Tool Physical Commonsense Question: "Do I have to put on oven mitts while taking out the turkey from the microwave after defrosting"
Answer: "If it is not too hot, you could also use a paper towel or wait for it too cool down"


2. Temporal Commonsense Reasoning Questions :
Description : A temporal reasoning question is one which involves an inherent understanding of time in order to come up with an answer. To this effect we have identified 3 key dimensions of temporal reasoning and illustrate each with an example

Duration Temporal Commonsense Question: "Can I quickly go grab bread from the nearby store while the Turkey is still in the oven?"

Ordering Temporal Commonsense Question: "Can I put the eggs to boil first, so they’re done by the time I peel vegetables?"

Frequency Temporal Commonsense Question: "How often should I stir the pot while the dish is being made?"
```

In total we have 6 categories :
1. Kitchen Tool Physical Commonsense
2. Ingredient Physical Commonsense
3. Duration Temporal Commonsense
4. Ordering Temporal Commonsense
5. Frequency Temporal Commonsense
6. None

The definitions of the 6 categories are as following:
1. Kitchen Tool Physical Commonsense:
Definition: This category involves reasoning questions about the physical attributes, functionality, or alternatives of various kitchen tools and utensils. The queries in this category require understanding the physical nature and practical utility of kitchen tools in various contexts. A common question might seek alternatives for a specific tool the inquirer lacks, like “What can I use instead of a whisk?” or understanding the capacity and limitations of kitchen utensils.

2. Ingredient Physical Commonsense:
Definition: Questions in this category necessitate understanding the physical properties, behavior, and alternatives of different ingredients used in cooking. It could also involve knowledge about how ingredients react under certain conditions or with other ingredients. Queries might include, “Can I substitute butter with olive oil?” or “What happens if I use baking soda instead of baking powder?”

3. Duration Temporal Commonsense:
Definition: This category deals with questions related to the time it takes for various cooking processes to complete. It involves understanding the duration required for different cooking methods and the timing necessary to successfully execute a recipe. Questions may involve the time required for marinating meat, baking a cake, or simmering a sauce, like “How long should I bake chicken breasts?”

4. Ordering Temporal Commonsense:
Definition: This category pertains to the sequence or order in which different cooking tasks should be performed for efficient and effective cooking. It involves understanding the chronological steps involved in preparing and cooking meals. Queries might relate to the order of adding ingredients to a dish or the sequence of cooking multiple dishes simultaneously, like “Should I cook the pasta before starting the sauce?”

5. Frequency Temporal Commonsense:
Definition: Questions under this category require understanding the frequency or repetition of specific cooking actions needed for preparing a dish. This could involve stirring, flipping, basting, or any other action that needs to be repeated at certain intervals for successful cooking. Common questions might include, “How often should I stir the risotto?” or “How frequently do I need to baste the turkey?”

6. None:
Definition: This category is reserved for queries that do not fit into any of the above-defined commonsense reasoning categories. These questions might not require specific physical or temporal reasoning related to cooking but may still be relevant to general cooking knowledge or advice. Queries like “What are some good side dishes for grilled chicken?” or “Can you recommend a wine pairing for spaghetti bolognese?” fall into this category since they don’t specifically involve physical or temporal reasoning.


For the question answer tuples given below, identify 1 or at most 2 categories to which the question belongs. The question may not belong to any of the categories, in which case it would belong to the `None` category.
Each question is accompanied with an answer and an alternate answer. We want to classify questions primarily based on the question text. You can use the answer and/or alternate_answer text to help with the question classification among the aforementioned 5 categories.

Present brief intermediate steps (if necessary) to arrive at your conclusion.


Sample Output :

'question': 'How long does the fermentation process take?'
'answer': 'The fermentation process takes about 2-3 days, depending on the temperature and humidity of your environment.'
alternate_answer : 'It depends on the temperature and the amount of sugar and yeast you use.  I usually let it ferment for a day or two, but it can take longer.  If you leave it in a warm place, it will ferment faster.  If you put it in the fridge, it will slow down the fermentation process.  I usually let it sit for a day or two, but it can take longer.  If you leave it in a warm place, it will ferment faster.  If you put it in the fridge, it will slow down the fermentation process.'
Reasoning:
Step 1. The question the amount of time to complete the fermentation process.
Step 2. Answer gives a timespan of 2-3 days.
Category: Duration Temporal Commonsense

Question Answer Tuples :
"""