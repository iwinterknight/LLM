prompt_4 = {
    "instruction": """Generate commonsense reasoning text for the sentence, question, answer and labels shown in the examples below:

Example 1.
    sentence : Max and Joey would often run through fields in a game of chase.
    question : How often do Max and Joey run?
    answer : 30 minutes
    label : 0   
    Expected Output :
        reasoning : run often does not imply every 30 minutes. it has a more general implication. 

Example 2.
    sentence : Max and Joey would often run through fields in a game of chase.
    question : How often do Max and Joey run?
    answer : 1.67 times a week
    label : 0
    Expected Output :
        reasoning : often does not mean that Max and Joey would run with exact frequency of 1.67 times a week 

Example 3.
    sentence : Max and Joey would often run through fields in a game of chase.
    question : How often do Max and Joey run?
    answer : every weekend
    label : 1
    Expected Output :
        reasoning : Max and Joey would run often in a game of chase suggests they would do this frequently and because they are children they could play over the weekends.  

Example 4.
    sentence : Max and Joey would often run through fields in a game of chase.
    question : How often do Max and Joey run?
    answer : on most weekdays before school
    label : 1
    Expected Output :
        reasoning : Max and Joey would run often in a game of chase suggests they would do this frequently and because they are children they could play before school began.
        
Example 5.
    sentence : Still , John vows to marry Mary if she meets him again. 
    question : How long had they known each other?
    answer : only a few minutes.
    label : 0
    Expected Output :
        reasoning : John vows to marry Mary if she meets him again suggests they have known each other a long time and share a deep connection.
        
Example 6.
    sentence : Still , John vows to marry Mary if she meets him again. 
    question : How long had they known each other?
    answer : 90 years.
    label : 0
    Expected Output :
        reasoning : People generally get married when they are 20 to 40 years old, sometimes older. 90 years is about as long as people live. John and Mary could possibly not be considering marriage after knowing each other for 90 years.
        
Example 7.
    sentence : Still , John vows to marry Mary if she meets him again. 
    question : How long had they known each other?
    answer : a few years.
    label : 1
    Expected Output :
        reasoning : John vows to marry Mary if she meets again suggests a deep connection grown over time between them. A few years is a plausible scenario for one to consider marriage.
        
Example 6.
    sentence : There is a grocery store in the neighborhood. 
    question : How long will it take for me to get groceries from the market ?
    answer : 1 month.
    label : 0
    Expected Output :
        reasoning : Visiting a nearby grocery store which is in walking or driving distance shouldn't take more than a few minutes to an hour.
        
sentence : There is a grocery store in the neighborhood. 
    question : How long will it take for me to get groceries from the market ?
    answer : a short while.
    label : 1
    Expected Output :
        reasoning : Visiting a nearby grocery store which is in walking or driving distance should take a short while.

sentence : There is turkey in the over. 
    question : Can I go visit my aunt in another city?
    answer : Yes you could.
    label : 0
    Expected Output :
        reasoning : Turkey in the oven takes a few hours to cook. Visiting a different city, depending how far it is, could take long. The turkey would get burnt and leaving the oven running could lead to fire hazard. 
    
sentence : There is turkey in the over. 
    question : Can I go on a vacation while it is cooking ?
    answer : No. The turkey will get burnt and the oven left running could potentially lead to a fire hazard
    label : 1
    Expected Output :
        reasoning : Turkey in the oven takes a few hours to cook. The vacation would last a few days atleast and so the turkey would get burnt and the oven left running could lead to a fire hazard.
        
sentence : There is turkey in the over. 
    question : Can I go visit a supermarket while it is cooking ?
    answer : Yes, if you have just begun cooking the turkey, it would take a few hours to roast. You can visit the supermarket but be back soon.
    label : 1
    Expected Output :
        reasoning : Turkey in the oven takes a few hours to cook. Visit to a supermarket should not take longer than a few hours at most. You would be in time to get the turkey out of the oven 
"""
}