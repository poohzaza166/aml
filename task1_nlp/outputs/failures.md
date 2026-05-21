# Task 1 — Curated failures of best model (D: word+char TF-IDF + LR)


Total errors: **232 / 1066** (21.76%)


Failure-mode tag counts (a sample may have multiple tags):


|             |   0 |
|:------------|----:|
| other       | 133 |
| contrast    |  50 |
| negation    |  38 |
| very-short  |  27 |
| intensifier |   1 |


## Most confident errors (top 12)

- pred=**0** true=**1** margin_pos=-0.96  tags=`negation`
  > neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway .

- pred=**0** true=**1** margin_pos=-0.96  tags=`negation,contrast`
  > the thing about guys like evans is this : you're never quite sure where self-promotion ends and the truth begins . but as you watch the movie , you're too interested to care .

- pred=**0** true=**1** margin_pos=-0.96  tags=`negation`
  > if you're down for a silly hack-and-slash flick , you can do no wrong with jason x .

- pred=**0** true=**1** margin_pos=-0.90  tags=`other`
  > byler is too savvy a filmmaker to let this morph into a typical romantic triangle . instead , he focuses on the anguish that can develop when one mulls leaving the familiar to traverse uncharted ground .

- pred=**1** true=**0** margin_pos=+0.89  tags=`other`
  > bravo reveals the true intent of her film by carefully selecting interview subjects who will construct a portrait of castro so predominantly charitable it can only be seen as propaganda .

- pred=**1** true=**0** margin_pos=+0.87  tags=`contrast`
  > mr . wollter and ms . seldhal give strong and convincing performances , but neither reaches into the deepest recesses of the character to unearth the quaking essence of passion , grief and fear .

- pred=**1** true=**0** margin_pos=+0.87  tags=`other`
  > whether quitting will prove absorbing to american audiences is debatable .

- pred=**1** true=**0** margin_pos=+0.87  tags=`very-short`
  > forgettable , if good-hearted , movie .

- pred=**0** true=**1** margin_pos=-0.87  tags=`negation`
  > while we no longer possess the lack-of-attention span that we did at seventeen , we had no trouble sitting for blade ii .

- pred=**1** true=**0** margin_pos=+0.87  tags=`other`
  > hilarious musical comedy though stymied by accents thick as mud .

- pred=**1** true=**0** margin_pos=+0.87  tags=`other`
  > a lightweight , uneven action comedy that freely mingles french , japanese and hollywood cultures .

- pred=**1** true=**0** margin_pos=+0.87  tags=`other`
  > if this is cinema , i pledge allegiance to cagney and lacey .
