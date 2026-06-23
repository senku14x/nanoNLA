# Base NLA vs length-penalty — held-out comparison

Same 1000 held-out prompts scored by every model; 100 examples shown. Metrics: **NMSE** = mse / predict-the-mean baseline (= 1 − FVE), **FVE** = fraction of variance explained, **tok** = AV's generated token count. Lower NMSE / higher FVE = better reconstruction; fewer tokens = shorter explanation.

## Aggregate (all common prompts)

| model | mean tok | NMSE | FVE | extraction |
|---|--:|--:|--:|--:|
| base | 126 | 0.468 | 0.532 | 99% |
| p0.0 | 127 | 0.415 | 0.585 | 99% |
| p0.001 | 92 | 0.408 | 0.592 | 100% |
| p0.002 | 74 | 0.430 | 0.570 | 100% |
| p0.006 | 32 | 0.518 | 0.482 | 100% |
| p0.015 | 25 | 0.550 | 0.450 | 100% |
| p0.03 | 14 | 0.775 | 0.225 | 100% |

## Examples

### Example 1
> _source ctx:_ Stacy Clark Named Tabitha TEAMember of the Year
Tabitha, Nebraska’s industry-leading Senior Care provider, congratulates Tabitha TEAMember of the Year, Stacy Clark. Clark is a continuum navigator in Tabitha’s central and western regions, which includes the cities of Grand Island and York, Nebraska.


| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 141 | 0.413 | 0.587 | Testimonial/review format signals continuation of positive patient praise about a local agency's services. The review follows a pattern of personal experience narrative with specific details.  Personal narrative momentum: the reviewer is mid-story, having established a problem (unanswered call) and  |
| p0.0 | 132 | 0.299 | 0.701 | Local news/feature article format: narrative profile of a community service initiative, establishing context for a personal story or testimony.  The phrase "impressed with her ability to make the complicated Medicaid services process understandable for families" sets up a quote or direct testimony f |
| p0.001 | 96 | 0.353 | 0.647 | Local business/community announcement tone, praising a healthcare nonprofit's commitment to the community.  Narrative momentum: the quote is mid-explanation of how Angelos was inspired to support Family Hope, requiring completion of the motivation/situation.  Final token "services" closes the descri |
| p0.002 | 72 | 0.440 | 0.560 | Narrative momentum: story building toward describing how the volunteer impacted the client's experience.  Direct quote continuation: the quote is mid-sentence, describing services received from the organization.  Final token "services" ends an incomplete thought mid-quote, expecting continuation exp |
| p0.006 | 32 | 0.387 | 0.613 | Narrative about program impact  explaining services provided  "services" closes sentence requiring continuation |
| p0.015 | 28 | 0.409 | 0.591 | community organization support services "educating families about behavioral health services" |
| p0.03 | 13 | 0.634 | 0.366 | and services |

### Example 2
> _source ctx:_ Stemmen van de Stad
"Stemmen van de Stad" is an audio-visual work that emerged from research by the Erasmus Initiative Vital Cities and Citizens. The work explores how different residents of Rotterdam perceive and experience their city. The project is an exploration of different perspectives, experi

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 127 | 0.520 | 0.480 | Narrative momentum from an exhibition catalog/description, establishing a thematic and visual analysis of the work "Buiten de stad," requiring continuation of critical commentary on the artwork's meaning.  Repetition/continuation of the phrase "outside the city" or the exhibition's name signals a re |
| p0.0 | 130 | 0.428 | 0.572 | Quotation continuation: the open quotation mark requires completion of the sentence about "living well in Amsterdam," which is mid-phrase.  Parallel structure established: "what people live with and what people live for" frames a broader participatory concept, suggesting the phrase "living well in A |
| p0.001 | 95 | 0.429 | 0.571 | Philosophical/reflective tone establishing conceptual depth about city living and identity formation.  The phrase "experiencing the" sets up a noun phrase completing the prepositional construction, requiring a specific concept like "city" or "metropolis."  Final token "the" is a definite article mid |
| p0.002 | 76 | 0.422 | 0.578 | Academic/institutional profile pattern describing research initiatives and collaboration outcomes.  The sentence is mid-thought, requiring completion of the research project description about residents of Rotterdam.  Final token "the" is a definite article beginning a noun phrase, almost certainly f |
| p0.006 | 29 | 0.480 | 0.520 | research project description plural noun reference required "of the" needs completion |
| p0.015 | 27 | 0.483 | 0.517 | research project description community perception survey "perception of the" |
| p0.03 | 13 | 0.525 | 0.475 | of the |

### Example 3
> _source ctx:_ Under leadership of new CEO, the chemistry technology company unveils strategic growth plan
OXFORD, England – OXECO, a chemistry technology company transforming product design and manufacturing, today announced the closing of a $10.5 million funding round secured by new Chief Executive Officer, Vass

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 110 | 0.215 | 0.785 | Business/venture capital announcement genre, establishing a funding event with company description, use of proceeds, and now investor details.  The text has shifted from company description to investor attribution, following the standard press release structure: announcement → deal terms → investor  |
| p0.0 | 131 | 0.196 | 0.804 | Financial/funding announcement genre: company press release format signals continued disclosure of investor details and transaction specifics.  The sentence beginning "It was led by London-based Laurus Capital Partners" introduces the lead investor, requiring completion of the funding round details  |
| p0.001 | 80 | 0.275 | 0.725 | News article structure transitioning from narrative summary to formal deal announcement, now detailing fund sources.  "Investment from Laurus Capital" signals the start of attributing financial backers, with fund/round specifics expected next.  Final token "Laurus Partners" is mid-sentence after "fr |
| p0.002 | 70 | 0.411 | 0.589 | Investment/financing announcement format establishes partnership details expected next.  The sentence "Backed by LSV Capital" signals investor attribution and deal structure details to follow.  "LSV Capital" is the final token, beginning a noun phrase requiring continuation of the institutional inve |
| p0.006 | 33 | 0.403 | 0.597 | funding announcement format investment round details expected "Led by Leumi Ventures" needs completion |
| p0.015 | 29 | 0.395 | 0.605 | investment deal announcement venture capital details "Led by Lendlay Capital" |
| p0.03 | 14 | 0.583 | 0.417 | Lincoln Partners |

### Example 4
> _source ctx:_ Shoppers wary of crowds and bored with the pandemic are increasingly filling their online carts, and EBay said it ended the fourth quarter of 2020 with 185 million active buyers, an increase of 7 percent.
EBay Inc. gave revenue and profit forecasts for the current period that

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 114 | 0.161 | 0.839 | Financial news reporting with analyst/revenue data structure: after stating Apple shares fell, specific financial guidance figures or comparisons are expected next.  The sentence "provided revenue and profit guidance that" sets up a clause requiring completion with specific forward-looking financial |
| p0.0 | 123 | 0.391 | 0.609 | Financial news report structure: earnings announcement follows standard format — revenue figures introduced, now expectations/outlook section expected.  Direct quote continuation pattern: the phrase "provided guidance that" is mid-sentence, requiring a noun clause describing investor expectations or |
| p0.001 | 79 | 0.368 | 0.632 | Financial news reporting establishing facts about earnings figures and revenue projections.  The sentence structure promises completion of what Amazon's guidance indicated, contrasting with investor expectations.  "that" is a relative pronoun ending an open clause, requiring an immediate verb phrase |
| p0.002 | 65 | 0.271 | 0.729 | Financial news article reporting Apple's earnings results and guidance.  Comparative contrast structure: "conservative guidance" sets up expected outcome versus market expectations.  The final clause "projections that" introduces a relative clause requiring a verb phrase describing guidance outcomes |
| p0.006 | 31 | 0.323 | 0.677 | Financial earnings report context comparison to analyst expectations "results that" requires completing clause |
| p0.015 | 23 | 0.436 | 0.564 | earnings report financial guidance "that" |
| p0.03 | 13 | 0.745 | 0.255 | results that |

### Example 5
> _source ctx:_ |FREE SHIPPING AVAILABLE ON THIS PRODUCT!*
* Please see our Terms & Conditions for details about Free Shipping Offers on selected products.
Hemavo2 Max by iForce Nutrition
PUMPS & ENDURANCE ENHANCEMENT:
HEMAVO2 MAX™ is the King of Pump for its unmatched Nitric Oxide boost for vasodilation combined w

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 133 | 0.227 | 0.773 | Product label format with standardized nutritional/usage instructions, following a strict formula supplement listing pattern (dosing, serving instructions, warnings).  The "Usage & Directions" section is mid-sentence, continuing a standard label formula instruction sequence that typically includes m |
| p0.0 | 120 | 0.276 | 0.724 | Product supplement instructions follow strict label/formulary conventions, requiring standardized directions language throughout. The text is mid-sentence in a preparation/mixing instruction.  The sentence structure "Mix in a [container]" signals a standard powder supplement mixing instruction, expe |
| p0.001 | 86 | 0.269 | 0.731 | Supplement product description format, with usage instructions following ingredient list conventions.  Shaking instruction in progress: "Mix in a" requires a container/shaker specification (e.g., "shake bottle" or "shaker cup").  Final token "a" is an indefinite article beginning a noun phrase speci |
| p0.002 | 72 | 0.330 | 0.670 | Product supplement label format requires standardized serving instructions and mixing directions.  The sequence follows a nutritional supplement label structure, with usage instructions now expected.  "Mix in a" is an incomplete phrase requiring a container noun (e.g., "shaker bottle," "blender") to |
| p0.006 | 32 | 0.431 | 0.569 | Supplement product description  mixing instructions pattern  "mix in a" requires container type |
| p0.015 | 26 | 0.457 | 0.543 | fitness supplement instructions preparation method "Mix in a" |
| p0.03 | 14 | 0.874 | 0.126 | Mix in a |

### Example 6
> _source ctx:_ How Logistics Leaders Are Using Wireless Charging
Charging for industrial electric vehicles like forklifts is a constant struggle. Just like charging for passenger and commercial electric vehicles, electric forklift charging relies primarily on plug-in charging, which is a time consuming task and ma

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 132 | 0.403 | 0.597 | List-continuation pattern: the text is enumerating companies leveraging AI (KPMG, Ernst & Young, McKinsey, Accenture, Capgemini, Deloitte, etc.) with "As a few" signaling an incomplete enumeration, strongly implying more names follow.  Argumentative momentum: the essay builds toward AI's business-va |
| p0.0 | 138 | 0.394 | 0.606 | Article follows a structured listicle format: introduction, benefits, challenges, now concluding with examples of AI adopters, signaling a list of company names is about to follow.  The narrative has built up to concrete real-world applications, and "a few" strongly implies a specific enumeration of |
| p0.001 | 80 | 0.387 | 0.613 | List/elaboration pattern: article introduces examples, building toward concrete company cases like Starbucks, Google, Microsoft.  Continuation of "a few" implies an incomplete enumeration of companies/organizations adopting AI sustainability tools.  Final token "a few" begins a list phrase requiring |
| p0.002 | 82 | 0.430 | 0.570 | Enumerated list structure with "A few" signaling continuation of company examples in the AI/ML adoption context.  Narrative momentum: article has introduced AI/ML benefits and is now providing concrete examples of adopters.  Final token "few" begins a quantified list of companies — next tokens must  |
| p0.006 | 37 | 0.254 | 0.746 | AI/tech article pattern  examples of AI adopters needed  "are just a few" requires completing the list |
| p0.015 | 22 | 0.501 | 0.499 | company examples AI advancements "a few" |
| p0.03 | 12 | 0.774 | 0.226 | A few |

### Example 7
> _source ctx:_ The Rams are getting good at this, wouldn’t you say? Bobby Wagner. Yeah, that Bobby Wagner. That’ll take a little of the sting out of losing Von Miller to the Bills. I know . Wagner is 31. But I also know he made a career high 170 tackles in what turned out to be his final season in Seattle.
I find 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 128 | 0.459 | 0.541 | Column-style sports journalism tone, blending analysis with humor ("homicidal maniacs," "stew in the cauldron") suggests continued witty commentary on NFL coaching hires.  The piece builds toward contrasting current coaching trends with past patterns, with each paragraph introducing a new angle or c |
| p0.0 | 133 | 0.958 | 0.042 | Conversational sports journalism tone, blending opinion, stats, and commentary on NFL roster decisions, establishing informal but knowledgeable voice.  The article transitions from discussing 53-man roster cuts to "The Monday Take," suggesting a new weekly segment/column format — a recurring column  |
| p0.001 | 85 | 0.359 | 0.641 | Sports news recap format with multiple topic transitions, suggesting another distinct sports observation follows.  Thematic momentum: each paragraph introduces a new football topic (Cowboys, Packers, Eagles, Steelers, Dolphins), implying continuation of similar observations.  Final period ends a Dol |
| p0.002 | 64 | 0.280 | 0.720 | Column format with numbered/listed football observations, continuing a sports journalism piece.  Each paragraph delivers a standalone insight, maintaining conversational but analytical tone throughout.  The final sentence ends a thematic list item about NFL positions, suggesting another observation  |
| p0.006 | 26 | 0.363 | 0.637 | Sports column style NFL draft analysis New topic transition expected |
| p0.015 | 24 | 0.513 | 0.487 | sports analysis NFL draft discussion "Another thought..." |
| p0.03 | 11 | 0.555 | 0.445 | _<extraction failed>_ |

### Example 8
> _source ctx:_ GZA has worked at numerous airports in New England on stormwater permitting related projects. GZA has prepared and peer reviewed SWPPPs under the NPDES MSGP program for many airports including Martha’s Vineyard Airport, Beverly Municipal Airport, Mansfield Airport, Westfield Barnes Airport, New Bedf

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 116 | 0.231 | 0.769 | Bulleted list of best management practices introduced with "including," requiring specific examples to follow — e.g., "stormwater separators," "permeable pavement."  Formal environmental/government reporting tone throughout, with technical vocabulary like "stormwater management," "Municipal Separate |
| p0.0 | 124 | 0.260 | 0.740 | Bulleted list pattern established: "best management practices," "included" signals an upcoming list of specific BMPs (e.g., permeable pavement, bioswales, retention basins).  Technical/environmental domain: stormwater management project document with formal municipal planning register, requiring pre |
| p0.001 | 88 | 0.256 | 0.744 | Technical/environmental report detailing facility upgrades at the USGS campus, following a structured problem-solution format.  The sentence is listing "best management practices," requiring specific examples of BMPs to complete the enumeration (e.g., stormwater detention, infiltration systems).  Fi |
| p0.002 | 54 | 0.252 | 0.748 | Ongoing list of BMP improvements being enumerated after "including."  Technical environmental/infrastructure reporting style with specific examples expected.  "including" is a list-introducing conjunction requiring concrete specific examples to follow immediately. |
| p0.006 | 31 | 0.296 | 0.704 | Wastewater infrastructure context specific mitigation measures expected "including" requires list continuation |
| p0.015 | 24 | 0.285 | 0.715 | stormwater BMPs specific practices "including" |
| p0.03 | 12 | 0.711 | 0.289 | including |

### Example 9
> _source ctx:_ “The Avengers” is adding another female face to its mostly male cast, and the four women up above are the current front-runners.
Morena Baccarin (“V”) and Cobie Smulders (“How I Met Your Mother”) are among those testing for the role later this week, according to The Hollywood Reporter. So are former

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 132 | 0.267 | 0.733 | Film production announcement pattern: news brief describing a sequel release date, following a predictable format of cast, director, release details.  The text is mid-sentence introducing Joss Whedon's role as director, establishing a new subject after the release date was stated — "Ralph Fiennes wi |
| p0.0 | 124 | 0.213 | 0.787 | News article about a superhero film, establishing factual reportage tone with cast and plot details, guiding continued factual/celebrity reporting.  The text has introduced the director ("Joss Whedon") and the actors, creating a pattern of introducing key figures with their associated roles or contr |
| p0.001 | 85 | 0.249 | 0.751 | Production/director information pattern: article profiles Star Wars: The Force Awakens, establishing J.J. Abrams's role now.  Narrative momentum: sentence begun with "J.J. Abrams is" requires completing his current role or action regarding the film.  Final token "is" opens a predicate requiring imme |
| p0.002 | 72 | 0.240 | 0.760 | Ongoing news article about Marvel movie updates, establishing production details in sequence.  Joss Whedon's role in the film is being introduced, requiring his position/title next.  "Joss Whedon is" — a predicate beginning requires an immediate noun phrase describing his role on the film. |
| p0.006 | 27 | 0.423 | 0.577 | film production details  directorial credits  Ralph Fiennes is |
| p0.015 | 27 | 0.313 | 0.687 | film production writer credit "Joss Whedon is" |
| p0.03 | 14 | 0.789 | 0.211 | Joshua is |

### Example 10
> _source ctx:_ 🎯 Transform into a trolley in 3 seconds, take everything you need with you wherever you go
Traffic congestion takes up 1% of Europe's GDP yearly, and parking alone can take up to 40% of driving time, 10% of CO2 emissions.
Personal mobility devices such as bicycle and cargo bicycle (with or without e

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 112 | 0.209 | 0.791 | Product description genre with structured marketing copy, introducing features and use cases in sequence.  The text has been building a case for a motorcycle with specific capabilities (e.g., obstacle avoidance, street riding), and a named product/feature "MOS C" is being introduced as the culminati |
| p0.0 | 160 | — | — | _<extraction failed>_ |
| p0.001 | 88 | 0.474 | 0.526 | Product announcement format establishes brand name "MOS C" as focal point requiring completion.  The review/sales pitch momentum builds toward elaborating MOS C's features and value proposition.  Final token "C" is mid-product-name ("MOS C"), immediately requiring continuation with the remaining let |
| p0.002 | 79 | 0.235 | 0.765 | Product launch announcement building toward naming/branding details.  The text is completing a marketing description, with "MOS C" clearly beginning a product name.  Final token "MOS C" is the truncated brand name mid-sequence, strongly constraining next tokens to complete the model designation (e.g |
| p0.006 | 28 | 0.353 | 0.647 | brand/product description format  model name continuation  MOS C[...] |
| p0.015 | 24 | 0.375 | 0.625 | product description company name "Mobility C" |
| p0.03 | 14 | 0.473 | 0.527 | MEC C |

### Example 11
> _source ctx:_ Universal Filmed Entertainment has taken a minority stake in the film and television production company headed up by Steven Spielberg.
Universal strengthens its relationship with Amblin Partners with its investment. The division of Universal Studios, in Universal City, had already been distributing 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 124 | 0.305 | 0.695 | Direct quote from Steve Wynn is mid-sentence, requiring grammatical and thematic completion — likely praising his involvement or legacy at Cirque du Soleil.  The celebratory/biographical narrative arc frames Wynn as a figure whose creative contributions and personal life are being honored, setting e |
| p0.0 | 115 | 0.392 | 0.608 | Formal press release/news format, with structured company announcement and executive quote block following standard PR conventions.  The quote from John Malone is mid-sentence, praising Turner Broadcasting System — the narrative momentum demands completion of his gratitude statement, likely referenc |
| p0.001 | 82 | 0.414 | 0.586 | Direct quote continuation: "The company that" requires completion describing Turner's legacy or Turner Broadcasting's role in the industry.  Celebratory tone and tribute structure: statement praising Turner's contributions and vision for the company's future.  Final token "that" opens a relative cla |
| p0.002 | 76 | 0.451 | 0.549 | Continuation of a speech quote attributed to Jon Favreau about the Disney+ studio.  The sentence structure "the studio that" requires a relative clause describing the studio's significance to Favreau.  Final token "that" opens a defining relative clause, demanding an immediate verb phrase continuing |
| p0.006 | 32 | 0.652 | 0.348 | celebrity quote structure career/legacy framing "that" relative clause requiring completion |
| p0.015 | 26 | 0.545 | 0.455 | celebrity quote studio success "the company that" |
| p0.03 | 14 | 0.694 | 0.306 | the company that |

### Example 12
> _source ctx:_ NARAHA, Japan (AP) — A drone nearly the dimensions of a slice of bread is Japan’s latest hope to get clearer footage of one of many reactors contained in the tsunami-hit Fukushima Daiichi nuclear power plant the place a whole bunch of tons of broken gas stay nearly 13 years after the catastrophe.
A 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 124 | 0.774 | 0.226 | Medical article explaining the coronavirus, establishing scientific rigor with detailed mechanisms of transmission. Formal informational tone maintained throughout.  Narrative momentum: the article has outlined transmission mechanisms (airborne/aerosol, contaminated surfaces) and is now expanding on |
| p0.0 | 119 | 0.917 | 0.083 | News article reporting on AI chatbot technology, establishing scientific findings about AI-generated content and user perceptions of authenticity.  Narrative momentum: the sentence is mid-clause describing what chatbots were able to do, specifically "able to" complete a capability, likely regarding  |
| p0.001 | 96 | 0.627 | 0.373 | Argumentative structure: article building case for why crypto should be regulated, with examples being enumerated.  Narrative momentum: "They can be capable" sets up a contrast explaining why crypto differs from traditional banks, implying limitations or capabilities.  Final token "capable" ends an  |
| p0.002 | 78 | 0.579 | 0.421 | Narrative momentum: explaining how some doctors are tackling vaccine resistance despite limitations.  Conditional "even though" clause establishes contrast between resistance and what physicians are capable of.  Final token "capable" is mid-clause, requiring a verb phrase completing what doctors can |
| p0.006 | 33 | 1.178 | -0.178 | Scientific reporting style explanation of research methodology "had the capability to" requires completion |
| p0.015 | 25 | 0.732 | 0.268 | vaccine effectiveness study research findings "were capable" |
| p0.03 | 14 | 0.914 | 0.086 | they had capable |

### Example 13
> _source ctx:_ Look, I tell you, lift up your eyes, and see that the fields are white for harvest (John 4:35). Soon after beginning his public ministry, Jesus turns to his small band of disciples and speaks these words. He goes out of his way to grab the attention of his listeners before making a simple statement.

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 132 | 1.227 | -0.227 | Structured religious devotion content with consistent verse-by-verse commentary pattern, moving through Psalms sequentially, establishing strong genre expectations for continued Bible commentary.  The devotional commentary format pairs verse references with personal, reflective commentary, maintaini |
| p0.0 | 138 | 1.159 | -0.159 | Sermonic/sermon content pattern: Bible verse, commentary, personal narrative, theological reflection — following a structured pastoral sermon format with citations.  Thematic focus on inner turmoil and trust in God: the passage addresses anxiety, conflict, and divine rest, requiring elaboration on s |
| p0.001 | 99 | 0.822 | 0.178 | Liturgical or devotional content pattern: Bible passages, hymns, and scripture references being listed sequentially.  Psalm 139 is the current scripture reference being cited, so verses or a question from that psalm is expected next.  Final token is "question?" — mid-sentence question from Psalm 139 |
| p0.002 | 84 | 0.796 | 0.204 | Biblical quote block is ongoing — a verse mid-completion requires continuation.  Psalm 130 is being cited, establishing the next line's scripture content.  The final token is an opening question mark mid-verse, demanding the corresponding biblical response to "Lord, why have you abandoned me?" — lik |
| p0.006 | 28 | 1.152 | -0.152 | Prayer quote continuation Biblical reference context "What troubles me?" |
| p0.015 | 35 | 1.427 | -0.427 | Biblical passage Psalm 130 "O my soul, what is my soul" |
| p0.03 | 15 | 1.430 | -0.430 | When am I discouraged? |

### Example 14
> _source ctx:_ There was a good article in the Globe & Mail recently outlining New Brunswick’s demographic challenges and how that is dampening the economic potential of the province. It certainly wasn’t a positive article – but it wasn’t the typical hatchet job either. It’s behind the paywall so I can’t link it h

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 138 | 0.509 | 0.491 | Ongoing survey methodology description: report is detailing survey parameters (sample size, recruitment, response rate), building toward demographic breakdowns and response categories.  Narrative momentum: the sentence ending "with first generation" is mid-clause, likely continuing demographic segme |
| p0.0 | 121 | 0.436 | 0.564 | Survey report presenting statistics about Toronto's ethnic/cultural diversity, maintaining informational and descriptive register throughout.  The text is mid-sentence completing a statistic about immigrant populations, listing categories of non-status/immigrant residents with "first generation" as  |
| p0.001 | 93 | 0.582 | 0.418 | Academic report style with formal policy implications section building evidence for language/immigration policy recommendations.  Continuation of demographic breakdown: the paragraph contrasts Toronto-born English speakers with immigrant English speakers, establishing pattern requiring completion of |
| p0.002 | 75 | 0.372 | 0.628 | Report findings being enumerated: statistics and demographic analysis about Scarborough's youth population.  Continuation of a list of Scarborough demographic data points about youth and immigration.  Final token "generation" begins a noun phrase describing Canadian immigration status; next tokens w |
| p0.006 | 30 | 0.583 | 0.417 | Survey methodology details  demographic breakdown in progress  "first generation" requires completion |
| p0.015 | 25 | 0.624 | 0.376 | demographic breakdown immigration statistics "first generation" |
| p0.03 | 13 | 1.147 | -0.147 | first generation |

### Example 15
> _source ctx:_ DNR spring egg taking operations ramp up
Now through mid-April, Minnesota Department of Natural Resources (DNR) fisheries staff will be working at lakes and rivers throughout Minnesota on the spring fish egg take to support the state's 17 hatchery operations.
Are you a newspaper subscriber but you d

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 100 | 0.430 | 0.570 | Formal government/municipal communications style maintained throughout, with institutional register for public service announcements.  Narrative momentum building toward a helpful resolution: after establishing the issue of missing library access and offering to help, a solution or contact instructi |
| p0.0 | 115 | 0.410 | 0.590 | Customer service/technical help desk register: formal yet reassuring tone advising users through a problem resolution process.  Narrative momentum toward resolving a login/user account issue: the user is being guided to fix an inability to access their account, with a clear call-to-action for custom |
| p0.001 | 76 | 0.399 | 0.601 | Customer support response mid-sentence, following a problem-reporting email with resolution steps being offered.  The sentence "will help you" sets up a directive completing the action of assisting the customer.  Final token "you" ends an incomplete clause requiring a verb phrase continuation like " |
| p0.002 | 65 | 0.302 | 0.698 | Customer service response guiding account access issue resolution, maintaining supportive tone.  The system is offering assistance with account setup or access after a purchase.  Final phrase "help you" strongly anticipates a verb completion like "set up your account" or "access your service." |
| p0.006 | 31 | 0.553 | 0.447 | Customer service reply context instructional/helpful tone "help you" requires completion |
| p0.015 | 24 | 0.572 | 0.428 | library help request customer service "help you" |
| p0.03 | 13 | 0.695 | 0.305 | help you |

### Example 16
> _source ctx:_ When I found out that my third sweet baby was going to be a girl, I couldn’t wait to play dress up. I had played dress up with my boys, and love the world of boy clothes. But they don’t make all the things for boys that they do for girls. The night I found out Lyla was coming, I started shopping the

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 118 | 0.322 | 0.678 | Product listing format for handmade jewelry, describing materials/origin of each item in a numbered list pattern (items 1–4 so far).  The text follows a consistent structure: product name, origin, materials, then detailed description — establishing expectations for completing item 4's description.   |
| p0.0 | 120 | 0.563 | 0.437 | Product description format for a children's jewelry brand, establishing style and origin — next content continues brand history/heritage.  Brand name "Hand" is being completed — "Karina Hand" likely refers to founder Karina Hand (a real person), so her last name continuation is strongly expected.  " |
| p0.001 | 98 | 0.284 | 0.716 | Product description listing brand and supplier details, establishing e-commerce retail register.  The sentence is mid-completion naming a supplier/partner brand "Lilah Hand [something]" requiring the product's full brand name.  "Hand" is the final token, the start of a proper noun brand name (e.g.,  |
| p0.002 | 64 | 0.306 | 0.694 | Product listing description building toward brand/manufacturer attribution.  List of curated handmade products being enumerated with origin details.  "Katie Hams Hand" is mid-sentence, strongly expecting a continuation like "Katie Hams Handmade" or similar brand name. |
| p0.006 | 36 | 0.433 | 0.567 | product description narrative  company/brand attribution pattern  "KA Hand" is the start of "KA Handmade" |
| p0.015 | 24 | 1.070 | -0.070 | product description brand name "Kris Hand" |
| p0.03 | 12 | 0.780 | 0.220 | Hand |

### Example 17
> _source ctx:_ Play now free mobile games on m.spellen.nl
The fast paced arcade game Fly With Rope is back! In Fly With Rope 2, you again use a rope to swing over the roofs of the coolest cities. Be careful not to kill the poor guy by releasing the rope at the wrong moment. Moreover, beware that the rope is elasti

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 119 | 0.227 | 0.773 | Promotional/informational app store listing pattern: repetitive "Download" call-to-action signals a template-style description block.  Game description narrative momentum: after listing features ("different animals, interesting situations"), a positive outcome or benefit is expected, completing the  |
| p0.0 | 122 | 0.306 | 0.694 | Game description genre following a structured promotional/entertainment format, with gameplay mechanics, features, and appeal already enumerated.  Narrative momentum toward completing the sentence about gameplay characteristics — "fun and" sets up a paired adjective or noun phrase (e.g., "fun and ad |
| p0.001 | 81 | 0.276 | 0.724 | Game description/marketing copy pattern establishes continuation of feature-highlighting language.  The phrase "fun and" sets up a paired conjunction requiring a complementary noun phrase (e.g., "challenge," "entertainment," "excitement").  Final token "and" is a coordinating conjunction mid-phrase, |
| p0.002 | 62 | 0.289 | 0.711 | Promotional description maintaining casual, enthusiastic tone throughout.  The sentence promises "fun and [something else]" continuing the list of gameplay qualities.  Final token "and" is a coordinating conjunction mid-list, requiring another adjective or noun completing "fun and..." |
| p0.006 | 33 | 0.354 | 0.646 | game description format  fun and entertainment promised  "and" requires a noun phrase completing the list |
| p0.015 | 23 | 0.315 | 0.685 | game description fun elements "fun and" |
| p0.03 | 13 | 0.488 | 0.512 | fun and |

### Example 18
> _source ctx:_ On Thursday in Sacramento, the Fish and Game Commission will hold the last big public comment hearing on a bold new plan to protect coastal waters off California’s north central coast. It promises to be a packed house and a lively debate, teeming with opinions as diverse as our state’s ocean wildlif

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 103 | 0.529 | 0.471 | Ongoing direct quote from Oregon governor Brown establishes a policy speech pattern, requiring continuation of formal political rhetoric.  The speech emphasizes balancing growth and preservation, suggesting next text will elaborate on shared Oregon values and future planning.  Final token "all" ends |
| p0.0 | 122 | 0.474 | 0.526 | Ongoing direct quotation from Oregon Gov. John Kitzhaber requires continuation of his speech's formal policy rhetoric.  The phrase "We all" opens a shared collective noun construction, typical of political speeches emphasizing bipartisan or universal agreement, e.g., "we all value/need/want" somethi |
| p0.001 | 109 | 0.657 | 0.343 | Ongoing argumentative structure: speech addresses coastal resource management, building toward shared community values.  Repetition/continuation pattern: "All of us and all of us" structure implies a parallel list of collective stakeholder values, e.g., "all of us care about the coast" or similar co |
| p0.002 | 66 | 0.421 | 0.579 | Ongoing direct quote from Rep. Greg Walden requires continuation in his voice.  Environmental advocacy speech building toward shared stakeholder concerns about the Klickitat Valley.  "We all" is mid-sentence, syntactically requiring a verb phrase expressing collective interest or benefit. |
| p0.006 | 24 | 0.493 | 0.507 | policy speech structure shared stake theme  we all |
| p0.015 | 23 | 0.504 | 0.496 | political debate community values "we all" |
| p0.03 | 13 | 0.670 | 0.330 | we all |

### Example 19
> _source ctx:_ 60 x 80 cm
Acrylic on canvas
Framed in white floating frame
Featuring Edmund de Waal (The hare with amber eyes)
This painting is part of the "Books on Canvas" Series. It is an interpretation of Edmund

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 133 | 0.553 | 0.447 | Book review genre following a structured evaluative framework (plot, characters, prose, recommendation), now moving to author bio/credits section. The pattern "This is a review... of..." signals formal review structure, with the author bio now introduced.  The sentence beginning "This is a review" t |
| p0.0 | 116 | 0.718 | 0.282 | Structured event listing format establishes context of describing an author's work/visiting.  The sentence introduces "Edmund" mid-name, requiring completion of the full name "Edmund D. Moreau" or similar, followed by an explanation of his work/relevance to the event.  Final token "Edmund" is the fi |
| p0.001 | 96 | 0.672 | 0.328 | Book/bibliography description genre with event promotion context, establishing a reader's guide structure.  The phrase "explore Edmund" promises a specific author introduction, requiring the full name and likely a descriptor or book reference to follow.  Final token "Edmund" is the beginning of an a |
| p0.002 | 90 | 0.511 | 0.489 | Book review/listing pattern established, requiring completion of bibliographic information about the author.  The review is building to describe Edmund De Waal's biography and work in relation to "The Hare with Amber Eyes."  Final token "Edmund" is the start of Edmund De Waal's full name, requiring  |
| p0.006 | 30 | 0.644 | 0.356 | Book review narrative  biographical introduction pattern  " Edmund" requires full name completion |
| p0.015 | 23 | 0.639 | 0.361 | book review Edmund "Edmund" |
| p0.03 | 13 | 0.636 | 0.364 | about Edmund |

### Example 20
> _source ctx:_ Mark Zuckerberg last edited by
A public institution for lending money to the poor at
a moderate interest, upon articles deposited and pledged; -- called
also mont de piete.
* These videos are coming directly from Youtube, they may or may not be most relevant to the word "Lom

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 122 | 1.079 | -0.079 | Blog/article aggregation format with labeled sections ("Latest," "Related Articles," "Most Viewed") repeating throughout, establishing a clear template for content listing.  List continuation pattern: "Most Viewed" section is mid-sequence, with multiple articles listed in descending order by views,  |
| p0.0 | 137 | 0.236 | 0.764 | Acronym completion: "Lom" is the beginning of "LoMo," the well-known loanword from the Japanese mobile lending service "Lending Loop."  The phrase "Lom-" signals a named brand/product introduction, requiring the full brand name ("LoMo" or "LoMo Money") followed by a description of how mobile lending |
| p0.001 | 122 | 1.264 | -0.264 | Financial journalism establishing a loan agreement between Lomé's city council and IFC, requiring continuation of the deal's terms and parties.  Narrative momentum: the article has concluded the deal's summary and is now transitioning to an expert analysis or quote, signaled by "According to Lomé-ba |
| p0.002 | 89 | 0.536 | 0.464 | Narrative shift to a new paragraph with "Last week, Lom" signaling a follow-up story about Lom.  Chronological storytelling pattern continues, building on the Lom loan program's expansion.  Final token "Lom" is mid-word or mid-phrase, beginning the proper noun "Lomahafo" or "Lom," requiring immediat |
| p0.006 | 34 | 0.326 | 0.674 | Financial case study narrative  Loan repayment process explanation  "Lom" is the start of "Lom" |
| p0.015 | 22 | 0.319 | 0.681 | loan description Lom "Lom" |
| p0.03 | 12 | 0.509 | 0.491 | Lom |

### Example 21
> _source ctx:_ Sneak Peeks for LEGO VIP Members
Posted on Thursday, February 7th, 2013 at 11:36pm by William, BZPower Reporter
An exclusive package has gone out to LEGO VIP members this month. In addition to a special VIP keychain minifigure, this package includes sneak peeks of upcoming sales and benefits at LEGO

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 133 | 1.354 | -0.354 | Numerical completion in progress: the text is mid-sentence describing the 40-point reward, requiring the full redemption threshold (e.g., "40 points to redeem").  Promotional/retail marketing copy style: formal brand messaging with specific figures ("80 points," "15000 points," "21000 points") drive |
| p0.0 | 126 | 1.479 | -0.479 | Structured list pattern: each entry has a price, percentage, and reward description, suggesting another entry or completion is expected.  The sentence beginning "As a member of..." is mid-clause, establishing a conditional/reward context typical of promotional content (e.g., "As a member of the loya |
| p0.001 | 88 | 0.926 | 0.074 | Promotional announcement pattern: details being listed (points, program name) continue with membership benefits.  Incomplete sentence structure: "only when you are a member of" requires completing the named loyalty program membership level.  Final token "a" begins an indefinite article mid-phrase, d |
| p0.002 | 95 | 1.283 | -0.283 | Promotional/marketing copy for British Airways ClubWorld program, maintaining persuasive tone throughout.  Continuation of a structured benefit listing, with each point building on membership advantages.  The final token "l" is the beginning of "l" in "l" — completing "becomes a full member" or "l"  |
| p0.006 | 32 | 1.021 | -0.021 | loyalty program benefits explained yearly points threshold pattern "to" begins completion phrase |
| p0.015 | 28 | 1.075 | -0.075 | loyalty program details membership tier "becomes a" incomplete |
| p0.03 | 16 | 1.262 | -0.262 | the "m" or |

### Example 22
> _source ctx:_ Maksim Mrvica, virtuoso pianist, winner of the MTV Music Awards and holder of several first prizes in prestigious international musical competitions, will perform at St. Michael's Fortress in Šibenik on July 1st!
After his tour in Australia and Asia, one of the most successful Croatian musicians wil

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 159 | 0.300 | 0.700 | Celebrity biography/promotional press release pattern establishes formal description tone throughout. The text follows a structured profile format introducing credentials and background.  Narrative momentum: a list of career achievements is being enumerated; the phrase "from the most iconic pop hits |
| p0.0 | 131 | 0.291 | 0.709 | Promotional/introductory music artist bio pattern: enthusiastic description of a new album release, building toward features and highlights.  The phrase "mixing some of the biggest" strongly implies a contrast or continuation — likely "hits" or "hit songs" to complete the well-known "mixing the bigg |
| p0.001 | 96 | 0.289 | 0.711 | Promotional/press release genre establishing artist credentials and tour context throughout.  The phrase "combine the biggest" sets up a list of musical styles/genres, requiring continuation naming specific elements (e.g., "hit songs," "international hits").  Final token "biggest" is mid-phrase, par |
| p0.002 | 75 | 0.275 | 0.725 | Promotional press release tone continues, describing the artist's live performances.  List of performance themes being enumerated: "classic, the most popular, and the biggest" signals an ongoing series.  "the biggest" is the final fragment, requiring a noun phrase describing hit songs or hits, e.g., |
| p0.006 | 28 | 0.384 | 0.616 | band description continuation  music genre focus  "the biggest" requires completion |
| p0.015 | 24 | 0.382 | 0.618 | musician biography career highlights "the biggest" |
| p0.03 | 13 | 0.626 | 0.374 | the biggest |

### Example 23
> _source ctx:_ Madura Coats Private Limited manufactures and distributes cotton and synthetic thread, yarn and industrial fabrics and operates as a subsidiary of Coats. The objective was to create an environment to experience and evaluate their special application categories like Apparel & Embroidery, Kite Flying 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 110 | 0.246 | 0.754 | Marketing/product description genre with promotional language ("excellent," "unique," "visually appealing") driving continuation of feature elaboration.  List-like enumeration of brand features: product description structure promises to continue detailing visual, functional, and sensory attributes o |
| p0.0 | 134 | 0.170 | 0.830 | Marketing case study format (product launch) establishes a pattern of describing brand strategy, objectives, and execution in sequence.  The sentence structure "The product features and benefits are" sets up a completion clause describing what these features/benefits *do* or *are communicated/expres |
| p0.001 | 84 | 0.221 | 0.779 | Marketing copy describing a branded experience, requiring continuation of feature elaboration about product benefits.  The phrase "product features are" sets up a predicate completing what features are doing — likely explaining their communication/impact.  Final token "are" opens a predicate clause  |
| p0.002 | 68 | 0.381 | 0.619 | List/continuation pattern: the sentence is building toward completing a description of product features.  Brand identity/identity language momentum: promotional retail description consistently highlights sensory and emotional qualities.  Final token "are" opens a predicate requiring completion descr |
| p0.006 | 30 | 0.441 | 0.559 | Product design context  value propositions being listed  "product features are" requires completion |
| p0.015 | 25 | 0.414 | 0.586 | product display concepts feature presentation "product features are" |
| p0.03 | 15 | 0.698 | 0.302 | the product features are |

### Example 24
> _source ctx:_ Search the Community
Showing results for tags 'subqueries'.
I am trying to calculate a metric to show the average percentage of incidents associated with changes by dividing 'the number of incidents linked to changes' by 'the total number of changes' and multiplying the result by 100. I would like t

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 156 | 1.044 | -0.044 | SQL query result pattern: structured query showing a joined table `ims_ticket_446556874180010806` with fields like `ims_workflow`, `ims_issuetypes`, `ims_ticket` — next tokens likely continue this table or output.  List continuation: "ims_ticket" appears multiple times in column names, suggesting a  |
| p0.0 | 121 | 0.801 | 0.199 | SQL query syntax in progress: a SELECT statement is being constructed, requiring the completion of the table name and field references.  Narrative context of data migration problem (restructure CMDB DB) drives expectation of technical field/column names like `ticket_id`, `incident`, or `description` |
| p0.001 | 99 | 0.689 | 0.311 | Technical documentation explaining an API/field mapping issue, establishing context for a specific solution.  The narrative is mid-sentence describing a table structure involving "ITSM" and "itilitsm," suggesting a table column name pattern continues.  Final token "itilitsm" is part of a table refer |
| p0.002 | 73 | 0.731 | 0.269 | Database/table structure being described, expecting column/table names and schema details.  Narrative momentum: the response is cataloguing database entities in an informational/help register.  Final token "itms" is part of the truncated table name "itms_incident," immediately requiring continuation |
| p0.006 | 35 | 0.509 | 0.491 | technical support context  database table structure explanation  "from ism_itsm" opens a table name |
| p0.015 | 26 | 0.808 | 0.192 | ticket system technical database "itil__itsm" |
| p0.03 | 14 | 1.026 | -0.026 | ita_itam |

### Example 25
> _source ctx:_ Opinion by Frank Short: Some weeks ago I wrote about the need for hearing aids for those in the Solomon Islands communities that have impaired hearing and hearing loss, especially affecting several young children.
In the United States I know of one or more charity organizations that donate hearing a

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 122 | 0.582 | 0.418 | Political policy advocacy speech pattern: formal legislative/proclamation language urging action with rhetorical devices and repeated emphasis on urgency ("urgent need," "imperative").  Narrative momentum toward justification: the "considering" clause opens a causal or conditional argument, requirin |
| p0.0 | 129 | 0.335 | 0.665 | Formal advocacy/legislative speech genre, using official register throughout — expects continued policy rationale and evidence-building.  The sentence ending with "considering" introduces a subordinate clause that must complete with specific factors explaining why health services are urgently needed |
| p0.001 | 83 | 0.338 | 0.662 | Argumentative momentum: speech building case for urgent malaria vaccine procurement, with reasoning being completed.  Causal clause continuation: "considering" introduces a specific reason/elaboration about the malaria threat's severity in the region.  Final token "considering" opens a subordinate c |
| p0.002 | 72 | 0.411 | 0.589 | Argumentative momentum: explaining why dental care is urgent for Tongan people.  Causal explanation expected: "considering" introduces supporting evidence for the problem's urgency.  Final token "considering" opens a subordinate clause requiring specific contextual reasons (e.g., pandemic, workforce |
| p0.006 | 32 | 0.575 | 0.425 | health advocacy momentum  covid-19 context required "considering" demands reason |
| p0.015 | 24 | 0.699 | 0.301 | public health concern government commitment "considering" |
| p0.03 | 13 | 0.641 | 0.359 | considering |

### Example 26
> _source ctx:_ Sensitivity Enhancement of Ammonia Gas Sensor based on hydrothermally synthesized rGO/WO3 nanocomposites
Authors: Deepak Punetha and Saurabh Kumar Pandey
Abstract: An ultrasensitive ammonia gas sensor based on hydrothermally synthesized rGO/WO3 nanocomposite with interdigitated chromium electrode ha

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 139 | 0.425 | 0.575 | Scientific abstract metadata format — fields like Journal, Volume, Authors, Abstract, Keywords, Pages follow structured academic conventions.  Bibliographic record listing expected: "Publishing Date" field requires a specific date (e.g., "2021", "2021-12-01") in a standardized date format, continuin |
| p0.0 | 120 | 0.640 | 0.360 | Academic paper metadata format, following structured journal article citation template with author list, abstract, keywords, journal details.  The "Publication Date" field signals a chronological record entry, establishing temporal context; a specific date string (day/month/year) is strongly expecte |
| p0.001 | 98 | 0.587 | 0.413 | Academic paper metadata table structure, with fields like "Abstract," "Citations," "Downloads," "Download Pdf."  Publication metadata pattern: after "Published Date," a specific date (e.g., "2023-08-15") or date format is expected next.  "Published Date" is the last field label in a metadata table r |
| p0.002 | 72 | 0.719 | 0.281 | Academic journal citation/abstract format establishes structured metadata fields.  Publication date section is a standard bibliographic field requiring a date value.  "Publication Date" is a label/header, directly expecting a colon and a date (e.g., ": 2024-03-25"). |
| p0.006 | 35 | 0.617 | 0.383 | Academic paper metadata format  Publication date field follows  "Publication Date" is a label requiring a date |
| p0.015 | 23 | 0.646 | 0.354 | academic paper publication metadata "Publication Date" |
| p0.03 | 13 | 1.369 | -0.369 | Publication Date |

### Example 27
> _source ctx:_ More than 60% of market drugs aimed at fighting disease target molecular proteins in the cell membranes – but techological limitations have previously limited researchers’ ability to observe these proteins in their natural environment.
Now, scientists in the lab of Yale School of Medicine’s Kallol G

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 134 | 0.361 | 0.639 | Scientific explanation momentum: the paragraph is detailing a technical finding about lipid removal from DNA samples, building toward a specific consequence or mechanism.  Parallel structure pattern: the sentence describes two simultaneous effects ("lipids are removed and their ability") implying a  |
| p0.0 | 117 | 0.304 | 0.696 | Scientific article explaining protein engineering technique, building toward a technical explanation of why purified proteins "lose their ability."  The sentence structure "lose their ability" demands a complement — a verb phrase describing what proteins can no longer do (e.g., "fold properly," "fun |
| p0.001 | 107 | 0.534 | 0.466 | Scientific explanation building toward a critical problem in vaccine delivery systems, contrasting success with limitations.  Parallel structure pattern: "while the nanovesicles successfully deliver the genetic payload and the ability" implies completion of what nanovesicles lack (e.g., "to replicat |
| p0.002 | 75 | 0.892 | 0.108 | Scientific explanation of protein aggregation challenges in synthetic cell research.  Logical argument building toward consequences of protein unfolding: loss of function follows "ability to..."  Final phrase "ability to" is an incomplete infinitive construction requiring a verb describing protein f |
| p0.006 | 29 | 0.915 | 0.085 | scientific explanation pattern  functionality of proteins  "ability to" requires completion |
| p0.015 | 22 | 0.491 | 0.509 | scientific explanation protein function "ability" |
| p0.03 | 15 | 0.789 | 0.211 | to retain their ability |

### Example 28
> _source ctx:_ On line journey company Journey.com produced a powerful debut in Hong Kong on Monday, with shares growing all around 4.55% from their challenge rate.
The China-based business now joins other U.S.-detailed Chinese tech heavyweights these types of as Alibaba,

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 120 | 0.246 | 0.754 | List-completion pattern: the article enumerates tech giants supporting China's economy; Tencent and Alibaba have been mentioned, strongly implying another company (e.g., Baidu, JD.com) follows as the series continues.  Narrative momentum toward economic optimism: the text builds China's economic res |
| p0.0 | 96 | 0.250 | 0.750 | Enumerated list of Chinese tech giants building data centers expected to continue (Alibaba just named, next entries like Tencent, Baidu, JD.com, or ByteDance are strongly implied).  Financial/business report genre establishing competitive landscape, requiring continued enumeration of Chinese compani |
| p0.001 | 83 | 0.319 | 0.681 | Enumeration pattern: "Alibaba Cloud, Tencent and Alibaba" lists competing Chinese tech firms, expecting at least one more example.  Market analysis context: discussing tech firms expanding to Southeast Asia, driving competition narrative requiring additional companies or elaboration.  Final token is |
| p0.002 | 79 | 0.209 | 0.791 | Continuation of a list of Chinese tech giants with Alibaba as the first item, requiring completion with additional names like Tencent.  The sentence structure "Alibaba, [others]" signals an ongoing enumeration of companies.  The final token is a comma after "Alibaba," mid-list, requiring an addition |
| p0.006 | 31 | 0.265 | 0.735 | Business news context competitor list continuation "Alibaba," requires another company name |
| p0.015 | 25 | 0.218 | 0.782 | tech stock list major companies "Alibaba," |
| p0.03 | 14 | 0.528 | 0.472 | Alibaba, |

### Example 29
> _source ctx:_ How Georgia Power Generates Electricity
Georgia Power has a number of different types of electric generating plants in its fleet. The diversity of fuels used in these plants enables us to provide a reliable power supply for our customers.
Traditional fuel sources such as coal, natural gas, hydro and

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 136 | 0.241 | 0.759 | Explanatory/informational pattern establishing a cause-and-effect explanation for the ash color problem, requiring completion of the removal process description.  Technical domain context: power plant emissions terminology ("air pollution controls," "scrubbers," "flue gas desulphurization") signals  |
| p0.0 | 142 | 0.178 | 0.822 | Technical/environmental report describing coal-fired power plant air quality processes, establishing factual prose style with precise terminology like "sulfur dioxide" and "fly ash."  Narrative momentum: the passage is explaining how emissions are controlled at the source, building toward completing |
| p0.001 | 85 | 0.187 | 0.813 | Industrial/technical description of coal plant emissions, establishing context for pollution control measures.  The phrase "removed by equipment" requires completion specifying what equipment or system (e.g., "located in the stack" or "at the plant").  Final token "equipment" ends an incomplete clau |
| p0.002 | 69 | 0.191 | 0.809 | Environmental/energy article explaining emissions controls at coal power plants.  Explanation of coal ash removal process requiring completion of the technical mechanism.  Final token "equipment" ends an incomplete relative clause ("by equipment"), requiring a prepositional phrase like "at the plant |
| p0.006 | 30 | 0.617 | 0.383 | Environmental/energy article context pollution removal explanation "equipment" requires completion |
| p0.015 | 26 | 0.473 | 0.527 | coal-fired plant description air pollutant removal "equipment" |
| p0.03 | 12 | 0.726 | 0.274 | equipment |

### Example 30
> _source ctx:_ Gardening has been a happy place for so many of us over the past few years, even those of us with no outdoor space. Undeterred, we’ve grown whatever we can indoors.
Among the most intriguing projects I’ve come across is growing an oak tree indoors. If you’ve never tried to germinate acorns, you must

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 129 | 0.907 | 0.093 | Numbered list format ("1. Choosing the right seeds," "2. Choosing the right container") signals continuation with item "3. Sowing your seeds," requiring sequential garden tips with consistent heading + body structure.  Practical gardening guide tone — conversational yet instructional, with step-by-s |
| p0.0 | 120 | 1.137 | -0.137 | Sequential gardening tutorial structure: each numbered/sectioned step explains a distinct phase in sowing peas, requiring continuation of practical instructions.  The text has been methodically covering preparation steps (soil preparation, sowing seed), implying the next logical step is germination, |
| p0.001 | 86 | 1.075 | -0.075 | Bulleted list structure continues, following the pattern of numbered steps; "Planting" signals the next step's description.  Cultivation guidance context requires practical, step-by-step gardening instructions for growing tomatoes from seeds.  Final token "Planting" is a section header/heading in th |
| p0.002 | 70 | 1.074 | -0.074 | Sequential gardening tutorial structure: numbered steps guide planting process step-by-step.  The process of planting is mid-explanation, with "Once your seeds are sprouted" signaling the next actionable step.  Final token "Once" opens a conditional clause requiring a verb phrase describing the next |
| p0.006 | 38 | 1.015 | -0.015 | Garden plant growing guide  Next step in transplant process  "Once you have your seedlings" begins procedural next step |
| p0.015 | 28 | 1.049 | -0.049 | planting instructions step-by-step process "Once you're ready" |
| p0.03 | 11 | 0.866 | 0.134 | _<extraction failed>_ |

### Example 31
> _source ctx:_ Consumers pay large sums for diamonds with perfect clarity, cut, color and carat. This technology capitalizes on the diamonds with imperfections or quantum defects. By measuring the spin properties of these defects, the lab can make drift-free precision sensors that are tied to fundamental physical 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 133 | 0.478 | 0.522 | News article structure signals a dateline/date stamp pattern: "October 22, 201" is a truncated year requiring 3 digits, almost certainly "201" → "2017" or similar modern year.  Formal science journalism register maintains objective, factual tone throughout, with institutional credit and technical te |
| p0.0 | 154 | 0.453 | 0.547 | Acronym expansion pattern: "XeTH" was spelled out as "X-ray-enabled Tilted High Harmonic" in parentheses, so "XeTH-" is the acronym being repeated to continue the sentence.  News article format with citation/award context: the text is describing a recognized scientific achievement, establishing a pa |
| p0.001 | 98 | 0.575 | 0.425 | Technical article about quantum metrology innovation, establishing credibility through citations and dates.  The article is concluding with a formal attribution citing the year of the breakthrough, requiring a specific four-digit year to complete.  "201" is a partial year mid-attribution, immediatel |
| p0.002 | 77 | 0.493 | 0.507 | Institutional news article transitioning to publication credit/citation format.  Pattern of listing technical details, patents, and research details before concluding with acknowledgment.  "The 201" is a truncated year (201X), almost certainly completing as "201X," requiring a 4-digit year to follow |
| p0.006 | 34 | 0.512 | 0.488 | scientific news article  publication citation required  "201" is year "2017" |
| p0.015 | 24 | 0.855 | 0.145 | scientific announcement date reference "111" |
| p0.03 | 14 | 0.860 | 0.140 | techno1 |

### Example 32
> _source ctx:_ Kristin Asbjørnsen Quartet:
(together with Thomas Dahl – guitars/vocals, Gjermund Silset – bass/vocals, Rune Arnesen – drums/Vocals and David Solheim – sound)
mars 7, 2014
Kristin Asbjørnsen Quartet will perform six concerts in Norway and Germany

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 133 | 0.378 | 0.622 | Biographical/professional CV/artist statement format establishes domain: career narrative with dates, institutions, and geographical movement.  Narrative momentum: the text has been detailing the artist's educational journey (USA → Germany → Austria) and is transitioning to career achievements, with |
| p0.0 | 118 | 0.249 | 0.751 | Press release format signals an upcoming list of dates/places, typical of tour announcements following "Germany" as a destination country.  Narrative momentum: the release has described the CD's contents, reception, and now concludes with tour logistics, requiring continuation of event details.  Fin |
| p0.001 | 97 | 0.209 | 0.791 | Biographical profile transitioning to performance tour details, establishing geographic scope of the project.  The phrase "Germany" ends a list of tour countries ("in Italy and Germany"), suggesting continuation with additional details like dates, venues, or further collaborators/performances.  Fina |
| p0.002 | 72 | 0.397 | 0.603 | Biographical/professional artist profile establishing career scope and geography.  List continuation pattern: "Germany" implies further country details or tour dates expected next.  Final token "Germany" is the end of a geographic scope list ("Germany"), likely followed by additional locations, date |
| p0.006 | 32 | 0.282 | 0.718 | biographical artist statement tour/venue details expected "and Germany" completes location list |
| p0.015 | 22 | 0.447 | 0.553 | tour announcement dates coming "Germany" |
| p0.03 | 13 | 0.685 | 0.315 | in Germany |

### Example 33
> _source ctx:_ I recently found that you can fix the hotplug problem (When you connect a USB device to the computer it doesn't activate automatically, you have to reboot), when using linux on HP Pavilion dv6220la.
In a previous post I wrote you have to add some flags to the linux kernel when you boot the computer.

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 129 | 0.284 | 0.716 | Technical documentation pattern: step-by-step guide for configuring IPv6 static routes on Debian/Ubuntu, maintaining instructional tone throughout.  Ongoing instruction momentum: the guide is mid-explanation of a method for adding multiple routes; the sentence beginning "The way I did that was to in |
| p0.0 | 117 | 0.306 | 0.694 | Instructional/tutorial genre with procedural steps being listed, promising a solution for enabling "aptitude" functionality in Debian/Ubuntu.  The text builds toward a specific software tool/package name, as "the package" promises a named product (likely a Debian/Ubuntu GUI or GUI installer package  |
| p0.001 | 80 | 0.321 | 0.679 | Technical instructions on configuring Ubuntu's clock settings, establishing context for the next step.  The phrase "install the package" signals an immediate software recommendation follows, specifying which package (likely `tzdata`).  Final token "package" ends mid-sentence after "install the packa |
| p0.002 | 80 | 0.296 | 0.704 | Technical how-to guide explaining Linux GRUB resolution for French locales  Step-by-step instructional momentum: the writer is mid-instruction giving solution details  "the package" ends mid-sentence, immediately requiring a specific package name (e.g., `grub`, `grub-legacy`, or `grub2`) to follow |
| p0.006 | 30 | 0.433 | 0.567 | technical installation guide context  software package reference "the package" requires name continuation |
| p0.015 | 24 | 0.476 | 0.524 | software installation Debian package "the package" |
| p0.03 | 13 | 0.728 | 0.272 | the package |

### Example 34
> _source ctx:_ Remarkable Trees Tour on Oct. 21 and 22
Oct. 14, 2008
- Five remarkable trees across Fairfax County are part of a free self-guided public tour on Oct. 21 and 22.
- Some of the trees are showcased in “Remarkable Trees of Virginia,” which features 120 great trees from across the state.
- Trees benefit

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 124 | 0.240 | 0.760 | Narrative momentum: article builds from local initiative to broader context, now transitioning to environmental/ecological benefits of urban trees.  Causal explanation pattern: "trees also help reduce air pollution and cut" sets up a paired list structure ("and cut...") requiring a parallel noun phr |
| p0.0 | 126 | 0.209 | 0.791 | Ongoing description of urban trees benefits: "cut" strongly anticipates a verb phrase like "cut down" or "cut down air pollution/air temperature," continuing the list of environmental benefits trees provide.  The article follows a news-press release format establishing urban trees' environmental val |
| p0.001 | 97 | 0.218 | 0.782 | Bullet-point list pattern with parallel structure: each item describes street trees' environmental benefits.  The text is enumerating ecological benefits of urban trees, so remaining benefits (air quality, carbon sequestration, heat reduction) are expected.  Final fragment "can cut" is mid-phrase, r |
| p0.002 | 81 | 0.220 | 0.780 | News article reporting on community tree-planting initiative, maintaining informational journalistic style.  List of benefits being enumerated: "save energy and cut" implies continuation of environmental advantages (e.g., "air pollution," "carbon emissions").  Final token "cut" ends an incomplete ph |
| p0.006 | 31 | 0.441 | 0.559 | environmental benefits list quantified pollution reduction "and cut" continues parallel structure |
| p0.015 | 25 | 0.396 | 0.604 | environmental benefits tree benefits list "can cut" |
| p0.03 | 13 | 0.863 | 0.137 | can cut |

### Example 35
> _source ctx:_ Well-differentiated neuroendocrine tumors (NETs) express high levels of the transmembrane somatostatin receptor subtype 2 (sstr2). Diagnostic and therapeutic radionuclides targeting this receptor have been successful in the clinic, resulting in the FDA-approval of 68Ga-DOTA-TATE for NET imaging and 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 159 | 0.517 | 0.483 | Scientific/technical paper abstract format: formal academic register established, expecting continuation of methodology and findings. The phrase "using 89Zr-labeled N-(2-hydroxypropyl)-β-cyclodextrin as an MRI-based imaging" signals a contrast or elaboration is coming.  Argumentative momentum: the a |
| p0.0 | 143 | 0.413 | 0.587 | Scientific paper abstract structure: the text introduces a study methodology (Gd/HPA-PEG nanoparticles) and its purpose, establishing the "methods" section framing of an abstract.  Domain-specific biomedical research context: nanoparticle labeling, MRI contrast agents, and tumor detection vocabulary |
| p0.001 | 96 | 0.374 | 0.626 | Scientific abstract structure: methods/results expected after stating purpose of radiotracer development.  Domain-specific terminology momentum: nuclear medicine imaging context requires continuation with methodology details like radiolabeling, tumor models, or imaging techniques.  Final token "imag |
| p0.002 | 69 | 0.477 | 0.523 | Scientific paper abstract structure: introduction of problem and methods signals results/findings next.  Narrative momentum toward experimental outcomes — the study's purpose drives expectations of results.  Final token "imaging" ends a clause introducing a goal phrase; immediate continuation likely |
| p0.006 | 35 | 0.427 | 0.573 | Research methods description  imaging application context  "for radionuclide imaging" ends mid-phrase |
| p0.015 | 26 | 0.509 | 0.491 | scientific study radiotherapy "radioligand imaging" |
| p0.03 | 17 | 0.759 | 0.241 | for radionuclide imaging |

### Example 36
> _source ctx:_ Poly aluminium chloride (PAC) , the white or yellow powder, is an inorganic polymer coagulant.Compared with traditional inorganic coagulants, fengbai poly aluminum chloride flocculation precipitation speed is fast, the PH value applicable range is wide, water purification effect is obvious and non-c

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 141 | 0.317 | 0.683 | Industrial chemical catalog/market report format, listing product descriptions, company names, and specifications for polymeric materials like PAM, PAC, and polyacrylates.  Repetition/continuation pattern: the text cycles through company name introductions, product descriptors, and technical specifi |
| p0.0 | 123 | 0.335 | 0.665 | Product description/chemical listing format: the text has shifted from article body to a bulleted product description listing chemical names, requiring continuation of this pattern.  List continuation pattern: several compound names have been listed with commas ("aluminum hydroxide, aluminum oxide,  |
| p0.001 | 93 | 0.308 | 0.692 | Chemical/nomenclature enumeration: listing synonyms for aluminum hydroxide continues with more compound names.  Structured database/metadata pattern: product listing with formula, synonyms, supplier info, and synonyms list suggests more synonyms follow.  Final token "aluminum hydroxide," is mid-list |
| p0.002 | 65 | 0.309 | 0.691 | Continuation of a list of aluminium compound chemical names separated by commas.  Database/FAQ content pattern with structured product listing and keyword enumeration.  Final token "hydroxide," is mid-list with a comma, strongly constraining next token to another aluminium compound name. |
| p0.006 | 33 | 0.421 | 0.579 | Chemical list continuation  product catalog format  "aluminum hydroxide," needs more chemicals |
| p0.015 | 28 | 0.337 | 0.663 | chemical names list inorganic compound "aluminum hydroxide," |
| p0.03 | 16 | 0.450 | 0.550 | aluminum hydroxide, |

### Example 37
> _source ctx:_ Construction of recycled water storage tank in Anthoupoli
The project consists of the construction of a 500 000m3 Water Storage Tank. It aims to collect water during the winter months which will be used for irrigation purposes during the summer months.
The Construction of the Anthoupolis Recycled Wa

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 125 | 0.146 | 0.854 | Project description follow-up: the text transitions from project scope to specific details, signaling enumeration of technical specifications or timeline information.  Structured project documentation pattern: formal tender/contract language with section headers like "Project duration" implies a lis |
| p0.0 | 122 | 0.142 | 0.858 | Project documentation structure: sections follow title/section header patterns, implying a table of contents or formal project summary format.  "Project duration" is a section header or field label, requiring an immediately following colon, number, or time expression like "Project duration: 12 month |
| p0.001 | 97 | 0.129 | 0.871 | Project specification document listing technical details, establishing pattern of labeled field entries like "Water pressure," "Water flow."  Narrative momentum: after detailing system components and pressures, duration/contractual timeline is next logical field.  "Project duration" is the final ent |
| p0.002 | 70 | 0.124 | 0.876 | Structured project report format with labeled sections driving content expectations.  List of project fields being enumerated sequentially (Location, Project scope, Project duration...).  "Project duration" is a section header, immediately requiring a specific time period/length response like "24 mo |
| p0.006 | 29 | 0.164 | 0.836 | Project description format  Project duration section header  "Project duration" needs completion |
| p0.015 | 22 | 0.272 | 0.728 | project description timeline "Project duration" |
| p0.03 | 13 | 0.483 | 0.517 | Project duration |

### Example 38
> _source ctx:_ Car Crash Compilation # 63
✔Like Us On Facebook http://www.facebook.com/CarCrashCompilationTV
All crashes in this video are non-fatal.Take this video as a learning tool.
This video is only for educational purposes. Be careful on the road. Drive
safely and keep yourself and others safe.
Watch Dash Ca

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 125 | 1.031 | -0.031 | Automotive enthusiast forum content mixing personal anecdotes with product reviews, establishing a pattern of casual, conversational tone with enthusiastic exclamations.  The text is mid-sentence introducing a specific car model example ("2020 Ford Mustang GT"), suggesting continued automotive conte |
| p0.0 | 135 | 0.467 | 0.533 | Repetition/continuation pattern: "Mustang" appears again, mirroring the earlier text's repeated noun usage — the series references "this particular Ford Mustang" and "the new Mustang."  Narrative momentum: the sentence is mid-completion, describing the car being reviewed ("this particular Ford Musta |
| p0.001 | 130 | 0.622 | 0.378 | The text is mid-sentence describing a 2022 Nissan GT-R NISMO's production specs, requiring completion of the technical claim.  The article follows a pattern of listing car features/specs with precise figures (e.g., "5.2 liters," "800 hp"), expecting similar specific data.  Final token "the 2" begins |
| p0.002 | 90 | 0.751 | 0.249 | List-style article continuing with Ford Mustang facts, establishing a factual enumeration pattern.  Incomplete sentence mid-thought: "They also own a particular 1" signals a model specification is being quoted.  The final token "1" is the beginning of a Ford Mustang trim/model number (e.g., "125 S"  |
| p0.006 | 34 | 0.795 | 0.205 | Car show review narrative specific model details expected "this particular car is a" requires model name |
| p0.015 | 25 | 0.736 | 0.264 | car review specific model name "this 1" |
| p0.03 | 14 | 0.992 | 0.008 | that 1 |

### Example 39
> _source ctx:_ Director Lokesh Kanagaraj’s new film Master features Vijay and Vijay Sethupathi in the lead roles. The film, which is produced by Xavier Britto, is off to a flying start at the box office in India and the international market. Master has taken a fantastic opening in Tamil Nadu and has managed to gra

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 142 | 0.392 | 0.608 | Celebrity gossip blog post format, with structured sections like "Read more" links and taglines establishing a repetitive, formulaic entertainment news article style.  A film production update is being detailed — "Kavithaiyum Karupattiyum" (K.K.K) has its cast listed, with two names given so far, st |
| p0.0 | 123 | 0.327 | 0.673 | Celebratory social media post pattern: upbeat tone, multiple cast mentions, song credits, and now a director credit following industry convention.  The director's name "Vijay Ram" is incomplete — a proper name mid-word or mid-credit requiring immediate completion, likely with surname followed by rol |
| p0.001 | 91 | 0.364 | 0.636 | Celebrity gossip/reality TV article structure signals an announcement of an upcoming collaboration with Vijay Ramesh.  The incomplete name "Vijay Ram" requires its suffix "esh" to complete, likely followed by details of the collaboration.  Final token "Ram" is mid-name in a collaboration announcemen |
| p0.002 | 68 | 0.399 | 0.601 | List of Tamil film names being enumerated sequentially after "Vijay, R Vijay and Vignesh"  Continuation of film director's name completion  "Vijay Ram" is a partial Tamil film title/producer name requiring completion of the full title name next. |
| p0.006 | 36 | 1.207 | -0.207 | Tamil cinema review context names completion expected "Vijay and Vijay Ram" requires full name |
| p0.015 | 30 | 0.261 | 0.739 | cast listing "Vijay and VJ Ram" "VJ Ram" |
| p0.03 | 14 | 0.989 | 0.011 | Vikram Ram |

### Example 40
> _source ctx:_ First Bank of Nigeria Limited (FirstBank) is Nigeria’s largest financial services institution by total assets and gross earnings. With more than 10 million customer accounts, FirstBank has over 750 branches providing a comprehensive range of retail and corporate financial services. The Bank has inte

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 93 | 0.282 | 0.718 | Job posting genre following standard HR format: qualifications, skills, then additional requirements — next section continues listing requirements/conditions.  Continuation of structured qualifications/requirements pattern: each bullet or clause introduces a new criterion (age, marital status, natio |
| p0.0 | 100 | 0.274 | 0.726 | Job posting listing qualifications, expecting continuation of requirement list with structured bullet points or phrases.  The "In addition" signals a list continuation pattern, implying more job requirements (e.g., skills, experience, education) are about to be listed alongside or after existing qua |
| p0.001 | 69 | 0.276 | 0.724 | Job posting format signals a list of qualifications/requirements continuing after "In addition."  Parallel structure expected: "In addition" mirrors the opening requirement list, implying further criteria follow.  Final token "In addition" is a transition phrase opening an additive clause, requiring |
| p0.002 | 47 | 0.394 | 0.606 | Job listing format with qualifications section continuing after "In addition"  List of required skills and attributes is expected to follow  "Additionally" signals continuation of comma-separated qualification criteria |
| p0.006 | 28 | 0.386 | 0.614 | Job description format  qualification criteria continuation  "Additionally" signals more requirements |
| p0.015 | 22 | 0.371 | 0.629 | job posting qualifications "In addition" |
| p0.03 | 12 | 0.654 | 0.346 | In addition |

### Example 41
> _source ctx:_ Please use this identifier to cite or link to this item:
LONG TERM PRESENCE OF ISLAND MASS EFFECT AT TIPUTA CHANNEL, RANGIROA ATOLL
|Title:||LONG TERM PRESENCE OF ISLAND MASS EFFECT AT TIPUTA CHANNEL, RANGIROA ATOLL|
|Contributors:||McManus, Margaret A. (advisor)|
Island Mass Effect
|Publisher:||Uni

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 122 | 0.919 | 0.081 | Academic journal metadata format, with structured fields (e.g., "Date: June 13, 2019", "Number: 1") establishing a bibliographic record template.  The copyright/reproduction restrictions notice follows a standard institutional pattern, typically ending with prohibited reproduction details like "any  |
| p0.0 | 125 | 0.938 | 0.062 | Academic metadata record format (bibliographic citation followed by repository permissions notice) dictates standard institutional repository language.  The phrase "distribution in" is a partial sentence in a copyright/repository permissions statement, strongly constraining next tokens to a preposit |
| p0.001 | 88 | 0.631 | 0.369 | Legal/academic citation boilerplate pattern establishing copyright permissions and restrictions on reproduction.  The phrase "distribution in" signals an incomplete list of prohibited reproduction methods (e.g., "any form or by any means").  Final token "in" is mid-phrase, requiring completion of th |
| p0.002 | 74 | 0.773 | 0.227 | Academic repository metadata format with copyright/distribution policy section established.  Legal distribution restriction language is standard in institutional repository terms.  "distribution in" is the beginning of a legal phrase, most likely completing as "distribution in any medium or format,  |
| p0.006 | 26 | 0.984 | 0.016 | Academic repository metadata  Reproduction policy statement  distribution in |
| p0.015 | 23 | 0.929 | 0.071 | academic repository distribution rights "distribution in" |
| p0.03 | 13 | 2.140 | -1.140 | distribution in |

### Example 42
> _source ctx:_ The Hikvision DS-2CE76U1T-ITMF is a 8 Megapixel (8MP) 4K Turret camera that incorporates Hikvision‘s EXIR infrared array technology. As a result, it provides industry-leading illumination in any low or zero-light environment.
Benefiting from an advanced, highly efficient light source that provides u

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 137 | 0.283 | 0.717 | Product specification list pattern: structured list of bullet points detailing technical parameters (e.g., "- 1920 × 1080p", "- 1.5 million pixels") establishing continuation format.  Domain-specific technical specs: each bullet follows a consistent pattern of "- [feature]: [value]" describing camer |
| p0.0 | 119 | 0.365 | 0.635 | Product specification document listing technical features of a smart camera, establishing a pattern of feature → value pairs with colon-separated values.  The document has enumerated several capabilities (night vision, intelligent recognition, motion tracking, remote monitoring, 3D imaging), suggest |
| p0.001 | 86 | 0.332 | 0.668 | Product specification list format, detailing technical features of surveillance equipment with bullet-style attributes.  Continuation of performance specifications — "Up to" begins a maximum value statement, requiring a numeric figure followed by a measurement unit (e.g., "1200m").  Final token "to" |
| p0.002 | 77 | 0.328 | 0.672 | Product specification list pattern: technical bullet points detailing camera features.  Numeric capability completion expected: "Up to" signals a maximum value like resolution or distance.  "Up to" is an incomplete phrase requiring an immediate quantified specification (e.g., "200m," "4K") to comple |
| p0.006 | 30 | 0.560 | 0.440 | Technical specification list maximum distance specification expected "Up to" requires distance value |
| p0.015 | 23 | 0.490 | 0.510 | technical specs camera features "Up to" |
| p0.03 | 13 | 1.174 | -0.174 | Up to |

### Example 43
> _source ctx:_ Precision profiling the breast cancer microenvironment
Prof V Speirs, Dr R Abu Eid
29 March 2019
The cancer microenvironment contains many different cell types, including various subsets of immune cells and cancer-associated fibroblasts, which support cancer growth.
Transcriptome Pattern Recognition

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 157 | 0.615 | 0.385 | Structured table of contents with hierarchical section headers (e.g., "1. The Cell, 2.1. Structure and Organisation, 2.2. Plant Virology") — a consistent two-digit/subordinate-numbering pattern dominates, predicting further subsection entries.  The document is a textbook index/reference list for Pla |
| p0.0 | 113 | 0.476 | 0.524 | Academic conference program listing format establishes a structured agenda pattern, with event details, speaker names, and session titles following a consistent format.  The list of speaker sessions is mid-enumeration; multiple named sessions with speaker names in parentheses are being listed, follo |
| p0.001 | 103 | 0.388 | 0.612 | Institutional/academic profile format continues, listing research groups and expertise of the John Innes Centre.  The sentence about "Research into Molecular Plant Path" is mid-phrase, requiring completion of the field name (likely "ology" or "ogenesis").  Final token "Path" is a truncated word in a |
| p0.002 | 71 | 0.385 | 0.615 | Academic conference announcement following a structured format with speaker bios and session details.  List of departmental or subject names continuing in sequence after "Microbial Plant Path"  Final token "Path" is the truncated end of "Microbial Plant Pathology," requiring immediate completion of  |
| p0.006 | 32 | 0.450 | 0.550 | Academic profile structure  Research interests listed  "Plant and Microbial Path" is truncated |
| p0.015 | 26 | 0.478 | 0.522 | academic profile research interests "Plant and Microbial Path" |
| p0.03 | 13 | 0.842 | 0.158 | Plant Path |

### Example 44
> _source ctx:_ Temperature Measurements of the Piston surface in a Research Compression Ignition Engine in Transient Conditions for 1d Model of Heat Transfer
Analysis of heat losses in internal combustion engines (ICEs) is fundamental to evaluate and improve the engine efficiency. Detailed and reliable heat transf

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 147 | 0.254 | 0.746 | Academic dissertation abstract structure: after introducing methodology and data, the text transitions to a discussion/contribution statement, signaling elaboration on thermal measurements' importance.  Narrative/argumentative momentum: the abstract builds from motivation (reliability importance) to |
| p0.0 | 124 | 0.248 | 0.752 | Academic thesis abstract establishing methodology context for measurement techniques; formal academic register maintained throughout.  Logical argumentative structure: the passage has introduced the research problem (temperature gradient modeling) and now transitions to justifying the study — measur |
| p0.001 | 87 | 0.392 | 0.608 | Academic paper introduction establishing motivation and research gap, requiring elaboration of measurement challenges.  The phrase "represent" signals an imminent predicate completing a noun clause about what surface temperature measurements mean/signify in combustion research contexts.  Final token |
| p0.002 | 66 | 0.263 | 0.737 | Academic thesis introduction establishing research scope and problem context.  The sentence sets up a contrast between known boundary conditions and unknown surface temperatures.  "The measurement of wall surface temperatures represent" is an incomplete predicate requiring a noun phrase or adjective |
| p0.006 | 31 | 0.580 | 0.420 | Academic paper structure  technical domain: combustion diagnostics  "represent" requires completing description |
| p0.015 | 28 | 0.451 | 0.549 | technical paper thermocouples "thermocouples represent" |
| p0.03 | 12 | 0.931 | 0.069 | represent |

### Example 45
> _source ctx:_ We need to challenge the assumption that fewer deer means more trees - Tom Turnbull
As a backdrop there’s the perhaps simplistic and poorly evidenced assumption that fewer deer will result automatically in more trees and an environment better equipped to combat climate change. But is this true – and

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 111 | 0.634 | 0.366 | Journalistic/informational article about women and mental health in construction, establishing a research-supported problem-solution structure throughout.  The article builds to a call-to-action section (comments box with prompt), signaling a closing engagement prompt to readers.  Final token "artic |
| p0.0 | 120 | 0.597 | 0.403 | Journalistic/news article structure with a "comment box" feature — the article is closing with a call-to-action prompt for reader engagement.  The phrase "Add your thoughts to this article" follows standard newspaper commenting platform conventions, typically followed by a link or button label like  |
| p0.001 | 86 | 0.579 | 0.421 | Journalistic/news article structure ending with a reader engagement prompt, signaling interactive commenting functionality.  The text follows a public health policy article with embedded call-to-action commentary, typical of policy journalism platforms.  Final token "article" closes the comment invi |
| p0.002 | 61 | 0.556 | 0.444 | News article structure nearing end: comment invitation typical of journalism platforms like The Guardian.  Formal news register with embedded reader interaction call-to-action.  Final token "article" closes a comment-disclaimer sentence, strongly suggesting a link or punctuation follows. |
| p0.006 | 29 | 0.577 | 0.423 | Call-to-action completion Comment submission prompt Final preposition "this article" |
| p0.015 | 29 | 0.678 | 0.322 | comment section read more link "Read more to comment on this article" |
| p0.03 | 16 | 1.207 | -0.207 | to comment on this article |

### Example 46
> _source ctx:_ ok so maybe someone with a little more handcent experience can help me with what i'm trying to do. when i first got handcent i'll admit i was hooked on the iphone bubbles, but 2 weeks later i guess i got that phase out of me, cause i switched to the

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 106 | 0.355 | 0.645 | User is sharing a personal device setup preference, contrasting two UI options, implying a positive recommendation is underway.  The structure "instead of [X] I use [Y]" signals a contrast favoring Android's default clock interface over the mentioned alternative (Faceroid).  Final token "the" opens  |
| p0.0 | 114 | 0.363 | 0.637 | User is describing a mobile app upgrade/downgrade process, establishing a before-and-after narrative about UI/UX changes on a touchscreen interface.  The phrase "I was on the" sets up a contrast requiring a specific UI element name — the default lock screen interface the user previously used before  |
| p0.001 | 100 | 0.336 | 0.664 | Personal tech blog style established, sharing a phone upgrade anecdote with casual, conversational tone.  Narrative momentum: the author has upgraded to Droid Eris, pivoted to talk about the lock screen, implying a contrast or transition from previous lock screen experience.  Final token "the" begin |
| p0.002 | 69 | 0.272 | 0.728 | User describing Android widget preferences, moving from one option to another.  Narrative pattern of listing tried widgets/options in sequence, implying a second choice being described.  "switched to the" is a transition phrase mid-sentence requiring a noun phrase naming the specific widget interfac |
| p0.006 | 30 | 0.554 | 0.446 | Smartphone customization context replacement feature explanation "the" requires a noun phrase |
| p0.015 | 25 | 0.389 | 0.611 | phone customization status bar widgets "changed to the" |
| p0.03 | 14 | 0.678 | 0.322 | now using the |

### Example 47
> _source ctx:_ **FINANCING AVAILABLE**BUILDING PERMIT IN HAND! San Bernardino County Approved plans for 1618 square feet of living space, plus an oversized 492 square foot garage, 113 square foot front porch, and a 372 square foot covered patio for a total of 2595 square feet are INCLUDED! Water meter has been ins

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 112 | 0.232 | 0.768 | Ongoing narrative review of a Las Vegas travel experience, listing sequential activities at the resort's golf course (putting green, practice range, driving range).  Repetition/continuation pattern: each activity described with action → outcome structure ("We played... We hit...").  Final token "P"  |
| p0.0 | 127 | 0.184 | 0.816 | Geographic/sport tourism description pattern: each location highlights amenities, attractions, and nearby points of interest, building a comprehensive profile.  Enumeration of local attractions around Pinal Lake is mid-list, following the established pattern of naming nearby locations with distance  |
| p0.001 | 88 | 0.120 | 0.880 | Travel blog narrative building toward describing activities at Parker Valley campsite near Palm Springs.  List/continuation pattern: "and P" begins a continuation of nearby locations or activities, likely "Palm Springs" or "Pioneertown."  Final token "P" is the start of a proper noun mid-phrase ("an |
| p0.002 | 81 | 0.193 | 0.807 | Trip review genre with personal travel narrative, listing destinations and experiences sequentially.  The review is enumerating stops and attractions on a multi-city trip, building toward a concluding activity.  Final token "P" is the beginning of a place name (likely "Petaluma" or similar Californi |
| p0.006 | 37 | 0.256 | 0.744 | trip review narrative momentum location description completing  "and P" is "and P" beginning "Pine Mountain" |
| p0.015 | 21 | 0.544 | 0.456 | trip review location names "P" |
| p0.03 | 11 | 0.665 | 0.335 | P |

### Example 48
> _source ctx:_ WGM was pleased to attend the 2018 Canadian Space Summit. WGM represented the mining and mine finance industries to the Canadian Government and aerospace industries. Both are seeking to build on Canada’s rich mining expertise by directing their expertise to space resource exploration and development

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 105 | 0.166 | 0.834 | Narrative momentum: article builds toward a forward-looking conclusion about Canadian companies leading in space mining/satellite technology.  Domain context: technical satellite/communications industry writing with policy implications, establishing Canadian leadership in emerging space domains.  Fi |
| p0.0 | 122 | 0.203 | 0.797 | Narrative momentum toward a forward-looking conclusion: the text has built a case for Canada's space industry strengths, now pivoting to a future prediction.  Domain-specific industry focus: the passage consistently centers on satellite technologies, space exploration, and Canadian innovation in spa |
| p0.001 | 93 | 0.137 | 0.863 | Narrative momentum: argument building toward Canada's leadership in space mining/commercial space, requiring a concluding assertion.  Domain context: space law and business strategy discussion drives expectation of forward-looking claims about Canada's role.  Final phrase "will likely be leaders in" |
| p0.002 | 75 | 0.208 | 0.792 | Argumentative momentum: article builds toward a concluding statement about Canadian leadership in space mining.  Thematic consistency: throughout emphasizes Canada's pioneering role and competitive advantage in space resource development.  Final token "in" opens a prepositional phrase requiring a no |
| p0.006 | 35 | 0.272 | 0.728 | expert quote continuation leadership in space mining "early leaders in" requires a noun phrase describing domain |
| p0.015 | 27 | 0.254 | 0.746 | Canadian space industry Canadian leadership "likely to be leaders in" |
| p0.03 | 16 | 0.335 | 0.665 | likely to be pioneers in |

### Example 49
> _source ctx:_ The What Athens bag fits everything! Sunglasses, city maps, a bottle of water, suntan lotion, cell-phone, hotel room pass..have you forgotten something?! Put a What Athens bag in your shoulder and start discovering the beauties of the city!
The Parthenon, the Acropolis, classical stadiums and statue

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 134 | 0.337 | 0.663 | Product listing format with structured specifications (name, price, description, details) signals a closing call-to-action section is beginning. The text follows a commercial retail page pattern.  The phrase "If you have any other questions about the product," is a standard CTA clause that almost al |
| p0.0 | 118 | 0.320 | 0.680 | E-commerce product listing format with specifications, materials, and usage instructions — a standard "if you have more questions" closing prompt is expected next.  The phrase "If you have any questions regarding the product" is a classic e-commerce boilerplate call-to-action, strongly implying a fo |
| p0.001 | 100 | 0.343 | 0.657 | Promotional/informational product page structure signals a closing call-to-action or contact inquiry section is underway.  The phrase "If you have any other questions regarding this product" sets up a conditional clause requiring completion with a recommended action (e.g., "please contact us" or "vi |
| p0.002 | 69 | 0.614 | 0.386 | Customer service/promotional webpage pattern with instructional tone throughout.  "If you have any questions about the product," sets up a conditional clause requiring a response action.  Final token "product," closes a conditional clause mid-thought, immediately expecting "please contact us" or sim |
| p0.006 | 26 | 0.344 | 0.656 | Product description format  Customer service transition  "For further information," |
| p0.015 | 29 | 0.570 | 0.430 | product description customer inquiry "if you have any questions about the product," |
| p0.03 | 14 | 0.484 | 0.516 | For more details, |

### Example 50
> _source ctx:_ Healthy teeth and gums are essential to enjoying everyday life. Taking good care of your oral health protects your smile. It also promotes better overall health. That’s because a lack of dental care can cause serious medical problems. Review six ways poor oral health can affect overall health.
1. He

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 133 | 0.444 | 0.556 | Enumerated list structure with bolded bullet points (e.g., "A weakened immune system") establishes a pattern of health-related risk factors. More risk factors are expected to follow this format.  Content domain signals: article systematically covers HIV/AIDS symptoms and risk factors, with the curre |
| p0.0 | 121 | 0.256 | 0.744 | Enumerated list structure: text has systematically covered HIV/STDs, heart disease, and cancer risks for people living with HIV; now completing the final section.  Medical/educational content pattern: each condition section presents a named health issue, explains its connection to HIV, and describes |
| p0.001 | 100 | 0.230 | 0.770 | Medical article structure: each disease section follows symptoms → transmission → treatments → complications pattern.  Content momentum: HIV section is mid-explanation, having stated "patients with HIV are more susceptible," requiring elaboration on specific oral health complications (e.g., gum dise |
| p0.002 | 75 | 0.217 | 0.783 | Informational article structure covering oral health topics systematically, with each paragraph introducing a new subtopic.  Narrative momentum: HIV-related oral health risks being introduced, requiring elaboration on infection impacts.  Final sentence ends mid-thought after "patients are more susce |
| p0.006 | 34 | 0.346 | 0.654 | Health article structure continuing  Explanation of HIV/AIDS impact on mouth  "AIDS weakens the immune system." |
| p0.015 | 31 | 0.850 | 0.150 | health article format oral health conditions "Individuals with HIV/AIDS are more susceptible" |
| p0.03 | 15 | 0.592 | 0.408 | Patients with HIV/AIDS. |

### Example 51
> _source ctx:_ Fans of the Buffalo Bills can all let out a collective sigh of relief.
After appearing to suffer what the Bills announced as a "chip fracture in his sternum," Bills wide receiver Jordan Matthews returned to practice on Sunday. Matthews wasn't dressed in pads meaning that he wouldn't partake in team 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 108 | 0.369 | 0.631 | Biographical narrative structure: article profiles Schuler's career, moving chronologically from college to NFL, so his current team affiliation is expected next.  The sentence begun with "He joined the" promises completion with Schuler's current NFL team (Bengals or another organization), requiring |
| p0.0 | 115 | 0.180 | 0.820 | Sports news article reporting on Buffalo Bills' 2016 NFL draft, establishing factual journalistic tone throughout.  Narrative momentum: the article has listed two key free agent additions (Smith, Davenport), and is now introducing their role/significance, following the structure of announcing and ex |
| p0.001 | 87 | 0.341 | 0.659 | Sports journalism narrative about NFL free agency, establishing background on Ryan's contract situation.  Continuation of Ryan's free-agent status explanation, requiring completion of who he was released from.  Final token "the" begins a noun phrase mid-sentence ("to the [team]"), immediately requir |
| p0.002 | 68 | 0.783 | 0.217 | Sports reporting context about a trade, requiring factual transaction details.  Narrative momentum: the trade just completed, now detailing consequences for the Bills.  "to the Bills" ends mid-phrase, immediately requiring the object — likely "Bills" or a new team's name. |
| p0.006 | 28 | 0.438 | 0.562 | Bengals trade context  contract/trade details  "to the" |
| p0.015 | 26 | 0.473 | 0.527 | NFL free agency Bills roster "signed to the" |
| p0.03 | 13 | 0.835 | 0.165 | and the |

### Example 52
> _source ctx:_ Google's ad-slinging juggernaut gobbles more BEEELLIONS in revenue
With no more Motorola to drag it down, sky's the limit for clicks and banners
Updated Google turned in another impressive earnings report on Thursday, with the giant ad-slinger's revenues reaching record highs both for the fourth qua

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 159 | 0.363 | 0.637 | Financial data pattern: the text lists year-over-year revenue figures, e.g., "up 16 percent to $47.3 billion," "up 6 percent to $13.7 billion," establishing a repetition template of percentage increase followed by dollar amount.  Narrative momentum: the revenue section has been systematically listin |
| p0.0 | 132 | 0.386 | 0.614 | Financial news article reporting on Apple's quarterly earnings, maintaining consistent pattern of presenting revenue figures with context and comparison data.  The sentence structure "Revenue was" sets up an expected numeric figure (likely in millions/dollars), consistent with the article's pattern  |
| p0.001 | 103 | 0.272 | 0.728 | Financial news report tracking Microsoft's Azure business performance, with consistent pattern of reporting revenue figures for each cloud segment.  Narrative momentum: the article has been detailing Azure revenue breakdowns, and "revenues were" signals the next specific financial metric (likely $X  |
| p0.002 | 67 | 0.315 | 0.685 | Financial report structure: revenue figures being presented sequentially across products.  Continuation of the "data centre, mobile" product breakdown pattern, each segment getting sales figures.  "revenues were" is an incomplete passive construction requiring a specific dollar amount or percentage  |
| p0.006 | 32 | 0.389 | 0.611 | financial reporting structure revenue figures expected "revenues were" requires a numeric value |
| p0.015 | 25 | 0.367 | 0.633 | financial reporting revenue figures "revenues were" |
| p0.03 | 14 | 0.402 | 0.598 | revenues were |

### Example 53
> _source ctx:_ Under the bankruptcy procedure the assets of Armavir Electrometallurgical Works LLC will be sold at an auction.
In particular, a site for steel manufacture will be sold.
The construction of the works has been carried out by the Industrial Union of Donbass since 2007. Investments were expected to hit

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 128 | 0.374 | 0.626 | News report structure following inverted pyramid: headline summary done, now detailed factual elaboration expected, covering construction status, financial delays, or government response.  Chronological/progressive narrative momentum: the text traces a timeline ("since 2009") and "design works were  |
| p0.0 | 127 | 0.258 | 0.742 | News article style reporting a Russian government decision, establishing factual/informational tone throughout.  The sentence ending "not proceeded beyond the design works" introduces a cause-effect explanation about the station's non-completion, strongly implying the reason or consequence will foll |
| p0.001 | 95 | 0.263 | 0.737 | News article reporting on Russian-Macedonian airport deal, establishing factual context about project delays.  The phrase "did not go any further than design works" signals an explanation of why the project stalled, implying further elaboration on consequences or timelines follows.  Final token "wor |
| p0.002 | 81 | 0.424 | 0.576 | News report structure: factual, third-person journalistic prose about stalled infrastructure project.  Narrative momentum: problem stated (delays), cause explained (design works halted), next expected detail about current status or consequences.  Final token "works" closes a clause explaining halt o |
| p0.006 | 28 | 0.499 | 0.501 | news article format  project development context  "design works" ends sentence |
| p0.015 | 23 | 0.539 | 0.461 | construction news contract details "design works" |
| p0.03 | 13 | 0.836 | 0.164 | design works |

### Example 54
> _source ctx:_ 27 Nov (NucNet): Installation of the core catcher for the Belarusian-2 nuclear power plant at Ostrovets in western Belarus will begin in March 2015 with the construction of both units at the site remaining on schedule, the operating company Belarus Nuclear Power Plant said.
The core catcher is alrea

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 124 | 0.456 | 0.544 | Ongoing news article about Ukraine's nuclear power plant expansion, establishing factual/infomatic register throughout. The article is detailing VVER reactor specifics and construction details.  Narrative momentum: the article has introduced the 600MWe units and is now providing technical specificat |
| p0.0 | 135 | 0.346 | 0.654 | Structured article format (headlines, summaries) signals factual, news-reporting style throughout — next text will continue this style with technical nuclear plant details.  The sentence "Both power stations, both" sets up an anaphoric repetition pattern, strongly implying parallel description of th |
| p0.001 | 85 | 0.333 | 0.667 | News article structure: factual reporting on nuclear power plant closure, establishing context about site and reactors.  List/intro pattern: "two reactors, both" sets up a description of reactor type, location, or characteristics (e.g., VVER design, Russian-built).  Final token "both" opens a coordi |
| p0.002 | 61 | 0.429 | 0.571 | News wire format reporting a nuclear plant shutdown, establishing technical detail context.  Specific factual continuation expected: VVER reactor type and location must be provided.  "both" ends the sentence, requiring immediate specification of reactor type(s) and their characteristics. |
| p0.006 | 31 | 0.561 | 0.439 | Nuclear plant context established technical specifications required "both" requires paired reactor details |
| p0.015 | 26 | 0.734 | 0.266 | nuclear plant details technical specifications "Both, both" |
| p0.03 | 14 | 0.938 | 0.062 | Both, both |

### Example 55
> _source ctx:_ The third story about dragoon Matthew Hervey. The series seems to get better each novel - no more simple, heroic adventures. Instead, the stories illustrate the complexities of the times (in this case circa 1817).
Hervey is also a thoughtful observer of what is going

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 127 | 0.428 | 0.572 | Historical narrative momentum: the scene sets up an investigation into a "great crisis" affecting the frontier, requiring its subject and cause to be named.  Literary/fictional register: prose novel describing character's thoughts ("what was going"), suggesting continuation in third-person limited o |
| p0.0 | 141 | 0.429 | 0.571 | Narrative momentum establishing social/political tensions in pre-revolutionary Ireland — the "What is going on" phrase signals a broad social or political question needs completion.  The phrase "What is going" strongly anticipates the word "on" (idiomatically completing "going on"), continuing the t |
| p0.001 | 80 | 0.302 | 0.698 | Narrative momentum: describing historical setting of 19th-century European rural life and social conditions.  Thematic focus: contrast between the sisters' privileged position and the social/political tensions around them.  Final phrase "what is going" ends mid-clause, requiring completion describin |
| p0.002 | 69 | 0.383 | 0.617 | Narrative momentum building toward describing the women's curiosity about the world beyond their home.  Thematic focus on female perspectives and expanding consciousness throughout the passage.  Final word "going" ends an incomplete clause "what is going," requiring immediate completion like "on" or |
| p0.006 | 44 | 0.303 | 0.697 | narrative momentum about societal change  historical context of late 1800s Irish emigration  "what is going" requires completion |
| p0.015 | 26 | 0.346 | 0.654 | social commentary narrative description "what is going" |
| p0.03 | 14 | 0.487 | 0.513 | what is going |

### Example 56
> _source ctx:_ CN C44-9WLs 2513 and 2503 have M348's train at 50mph approaching Henry House. With the rapid rebuilding of CN's Dash 9 fleet, it's a wonder how many more opportunities one will have to shoot a pai... (more)
The first revenue trip, leading, for Metra's rebuilt SD70MACHs as 2135 arrives at Long Lake.


| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 123 | 0.790 | 0.210 | Photo caption series pattern: short captions describe rail scenes with visual details, establishing rhythmic, evocative prose style throughout.  Narrative/visual continuity: each caption builds a distinct rail scene; the next entry likely describes another unique train, track, or station scenario, p |
| p0.0 | 113 | 0.587 | 0.413 | Photo caption series pattern: each caption describes a specific railway scene with location and subject detail, suggesting another descriptive caption follows.  Narrative momentum of railway photography blog: captions alternate between locomotive types (Amtrak, Canadian, Amtrak), suggesting continua |
| p0.001 | 92 | 0.273 | 0.727 | Photograph caption series pattern: each line describes a distinct railroad image with brief commentary, continuing sequentially.  Narrative/visual content: each caption introduces a new scene — engine, tracks, yard — establishing momentum toward another rail-related image description.  Final token e |
| p0.002 | 63 | 0.298 | 0.702 | Catalogue/photography series pattern: each entry is a new photo with description.  Continuation of observational railway imagery with consistent descriptive caption style.  Final token ends a sentence describing rail maintenance activity; next entry likely introduces another distinct railway scene o |
| p0.006 | 31 | 0.833 | 0.167 | Photo caption sequence pattern  railway photograph gallery context  "painted undercarriage" |
| p0.015 | 25 | 0.792 | 0.208 | rail photography captions date/location context "empty train" |
| p0.03 | 11 | 0.491 | 0.509 | _<extraction failed>_ |

### Example 57
> _source ctx:_ You might realize that you whenever you go home for break – winter, summer or spring – you always seem to run into people you went to high school with. Whether you come from a small town or a big city chances are you still end up seeing someone you know. For some people that may be a great thing, bu

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 134 | 0.854 | 0.146 | Argumentative momentum: essay builds a contrast between two parenting styles, establishing a framework of comparison for subsequent discussion of modern vs. traditional approaches.  Repetition/continuation pattern: the "then versus now" framing is explicitly set up; "then" and "now" are promised but |
| p0.0 | 134 | 0.384 | 0.616 | Narrative/argumentative momentum: essay explains peer pressure origins in adolescence, building toward how it affects personal identity and social status.  Continuation of parallel structure: the sentence lists how peers influence the narrator ("what your life is like," "what you're doing"), implyin |
| p0.001 | 90 | 0.316 | 0.684 | Ongoing narrative argument about peer relationships evolving beyond envy to genuine curiosity and connection.  Continuation of a list/elaboration pattern: the text is mid-sentence explaining what friends are now interested in discovering about each other.  Final token "doing" ends an incomplete clau |
| p0.002 | 75 | 0.388 | 0.612 | Narrative momentum building toward a personal reflection on social comparison and self-doubt  List of social media pitfalls is being elaborated with specific examples and consequences  Final clause "what you're doing" ends mid-thought, expecting completion about how others' lives are judged or contr |
| p0.006 | 32 | 0.484 | 0.516 | blog post narrative voice  social comparison theme continuing  "what you're doing" requires completion |
| p0.015 | 26 | 0.546 | 0.454 | first-person narrative relationship observation "what you're doing" |
| p0.03 | 15 | 0.860 | 0.140 | what you're doing |

### Example 58
> _source ctx:_ Both myeloablative and reduced-intensity conditioning regimens used prior to hematopoietic cell transplantation (HCT) will cause some degree of post-transplant immunodeficiency in recipients. In addition, both chemotherapy- and radiation-based conditioning regimens can cause organ and tissue damage.

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 144 | 0.422 | 0.578 | Academic medical review article on chemotherapy toxicity, establishing pattern of defining then characterizing each toxicity category systematically.  The text is mid-sentence defining "mucositis/mucos" — clearly truncating "mucositis," requiring completion of this term before elaborating on its pat |
| p0.0 | 135 | 0.487 | 0.513 | Medical review paper on mucositis treatment, establishing a systematic structure where each section covers a category of treatments with evidence-supported findings.  The text has followed a consistent pattern: introduce treatment category, list supportive evidence, cite studies, then summarize outc |
| p0.001 | 96 | 0.392 | 0.608 | Medical/clinical content establishing evidence-based treatment guidelines for radiation therapy mucositis management.  Continuation of a structured list of "oral mucositis treatment options" with prior examples ("mouthwash, oral gel") setting expectation for another method.  Final token "oral muc" i |
| p0.002 | 72 | 0.366 | 0.634 | Medical article following clinical content structure, moving through treatment sections systematically.  The section "Management of oral mucositis" is being introduced, signaling clinical guidance ahead.  "oral muc" is a truncated word, almost certainly completing as "mucositis," continuing the head |
| p0.006 | 31 | 0.640 | 0.360 | Medical clinical trial description  Symptom-focused enumeration  "oral muc" is partial word |
| p0.015 | 25 | 0.389 | 0.611 | clinical presentation oral mucositis "oral muc" |
| p0.03 | 13 | 0.740 | 0.260 | oral muc |

### Example 59
> _source ctx:_ We are Accountable to God
Romans 14:12 So then every one of us shall give account of himself to God
2 Corinthians 5:10 For we must all appear before the judgment seat of Christ; that every one may receive the things done in his body, according to that he hath done, whether it be good or bad.
We are 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 141 | 0.767 | 0.233 | Biblical quote in progress: the model must complete the verse "enter in and go out, and feed your sheep" from John 10:9, continuing the parallel "enter in" structure.  Theological/expository context: the entire passage is a gospel commentary, building from a historical example (Hagar/Sarah) toward a |
| p0.0 | 135 | 0.600 | 0.400 | Biblical/devotional register: theological exposition with scriptural citation ("Ephesians 2:17-18") establishing context for religious content.  Scriptural continuation momentum: the quotation is mid-sentence, requiring completion of Jesus' words about "going in and out freely" — a well-known passag |
| p0.001 | 116 | 0.490 | 0.510 | Religious/evangelistic content explaining church growth in Africa, establishing a contrast between African believers' spiritual experience and outsiders' perception.  The quote from Christ is mid-sentence, describing the privileges of sheep entering God's fold, requiring completion of the biblical p |
| p0.002 | 77 | 0.582 | 0.418 | Biblical exposition structure: passage analyzed, now transitioning to supporting scripture.  Repetition/continuation of Jesus' words from John 10:9 about entering through Him.  Final token ends mid-sentence after "free to go in and out," requiring the closing words of that verse: "and find pasture." |
| p0.006 | 35 | 0.882 | 0.118 | Biblical scripture citation expected  Scripture completion required  "and go out," follows standard Gospel verse |
| p0.015 | 27 | 0.925 | 0.075 | Biblical quotation freedom motif "enter in,out" |
| p0.03 | 15 | 1.547 | -0.547 | and go in, |

### Example 60
> _source ctx:_ Progress continues on the new Interstate 85 bridge that will span the Yadkin River between Davidson and Rowan counties.
Progress continues on the new Interstate 85 bridge that will span the Yadkin River between Davidson and Rowan counties. Crews have recently began fitting girders along the eventual

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 103 | 0.584 | 0.416 | Direct speech continuation: a contractor's quote is mid-sentence, requiring completion of his ongoing thought about project status.  Narrative momentum: the sentence describes the project nearing completion, implying optimistic forward-looking language about finishing timelines or milestones.  Final |
| p0.0 | 123 | 0.234 | 0.766 | Direct quote in progress: an official's speech is being reported, requiring completion of the spoken statement's thought.  Sequential construction milestone narrative: "we have to do our east side, then the south side...and we" signals a list of remaining tasks still in progress, requiring continuat |
| p0.001 | 77 | 0.246 | 0.754 | Direct quote from Jim Langer is ongoing, requiring continuation of his speech about project timeline.  The sentence structure "and we" sets up a paired completion about what remains to be done after the bridge work.  Final token "we" is mid-sentence within the quote, demanding immediate completion d |
| p0.002 | 79 | 0.315 | 0.685 | Direct speech continuation: Bob's quote is mid-sentence, requiring completion of his thoughts.  Construction project context: the narrative follows a progress timeline with specific milestones mentioned.  Final token "we" begins a first-person plural verb phrase, most likely continuing with a future |
| p0.006 | 30 | 0.419 | 0.581 | Construction update narrative  Project completion timeline framing  "and we" opens coordinated clause |
| p0.015 | 23 | 0.410 | 0.590 | construction progress schedule update "and we" |
| p0.03 | 13 | 0.661 | 0.339 | and we |

### Example 61
> _source ctx:_ Turquoise Geometric Doormat
This product is currently sold out.
Strong fibers made from the husks of coconuts set the tone for a sustainable home. This durable doormat features a bold geometric pattern in a versatile muted turquoise color making it the perfect addition to any front door.
- 18 x 30 i

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 131 | 0.307 | 0.693 | Product specification list format established: structured bullet-point attributes being enumerated for a product page, expecting material composition details next.  Material specification pattern: fabric composition percentage expected after "100" (mirroring earlier mention "100% polyester," strongl |
| p0.0 | 128 | 0.272 | 0.728 | Product description format established: e-commerce listing style with fabric specs following bullet points, requiring continuation of technical specification details.  Percentage completion of a percentage composition — "100%" strongly signals a textile composition term like "100% cotton" or "100% p |
| p0.001 | 93 | 0.285 | 0.715 | Product description genre with material/specs listing, following e-commerce bullet-point format.  Material composition details expected: "%100" signals a percentage composition phrase, e.g., "100% organic cotton" or "100% recycled polyester."  The final token "100" is the start of a percentage figur |
| p0.002 | 67 | 0.789 | 0.211 | Product specification format following structured material/quality bullet points.  List of percentage-based fabric composition details expected to continue.  "100%" is the start of a percentage specification, immediately requiring a fabric type or material name (e.g., "100% Cotton"). |
| p0.006 | 29 | 0.263 | 0.737 | Product description format material composition expected "100" begins percentage |
| p0.015 | 24 | 0.318 | 0.682 | product description material composition "100" |
| p0.03 | 14 | 0.839 | 0.161 | 100 |

### Example 62
> _source ctx:_ Originally Posted by pvaudio
This corkscrew spin business is not easy to understand because you cannot picture it easily. Here is the best way to do so, IMO. Either imagine or put a tennis ball in front of you. There are 3 directions: x (left and right), y (forward and backwards) and z (upwards and 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 119 | 0.201 | 0.799 | Forum reply responding to a request for a "visual of pitch angles on a missile," establishing context of a military/munitions forum post.  The author has been building toward providing a specific reference or image, having introduced "the diagram shown below" with a link, suggesting they will now de |
| p0.0 | 140 | 0.189 | 0.811 | Technical/educational aviation discussion explaining aircraft reference frames (earth-fixed vs. aircraft-fixed) in context of torque vectoring and aerodynamics.  The conversation is building toward defining a specific technical diagram or coordinate system (likely Euler angles, yaw/pitch/roll axes,  |
| p0.001 | 96 | 0.244 | 0.756 | Technical explanation of aviation mechanics with consistent diagrammatic/descriptive illustrations throughout.  The phrase "This is" sets up an explanatory clause describing the figure or concept just shown, requiring a descriptive noun phrase or elaborating statement.  Final token "is" opens a pred |
| p0.002 | 61 | 0.225 | 0.775 | Scientific explanation of three-axis coordinate system with consistent diagram-description pattern.  Continuation of a detailed description of a visual diagram or projection setup.  "This is" ends an incomplete sentence introducing what the coordinate system/diagram represents or illustrates next. |
| p0.006 | 25 | 0.328 | 0.672 | Technical aerospace explanation Reference to coordinate system "This is" |
| p0.015 | 22 | 0.335 | 0.665 | technical diagram coordinate system "This is" |
| p0.03 | 12 | 0.473 | 0.527 | This is |

### Example 63
> _source ctx:_ It was … a Rock Monster. Naturally, this movie takes place out in the wilderness (somewhere in Europe, as a matter of fact) … and there is a reference to King Arthur’s legendary Sword in the Stone. I love that Disney cartoon.
But be forewarned!! Some quiet afternoon while watching TV, you, too, may

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 160 | 0.323 | 0.677 | Narrative setup establishing a relatable "we may all do this" scenario, inviting readers to connect with the author's experience. The second-person address creates a direct community appeal, suggesting the next text continues building this personal anecdote into a cautionary or comedic point.  The c |
| p0.0 | 135 | 0.315 | 0.685 | Humorous/delightful narrative tone established throughout, building toward a funny or amusing observation about viewers reacting to the film's content (the film is about a woman in a wheelchair who finds humor in mundane life).  The rhetorical structure "Just because you saw this movie on TV... you  |
| p0.001 | 91 | 0.520 | 0.480 | Humorous/ironic setup establishing a relatable scenario about mind-wandering, requiring a punchline payoff.  List of "what you may" implies continuation of hypothetical consequences or outcomes of daydreaming.  Final token "may" opens a modal verb construction, immediately requiring a verb phrase li |
| p0.002 | 76 | 0.430 | 0.570 | Humorous, conversational tone established throughout — light, playful humor expected to continue.  Narrative momentum building toward a relatable personal anecdote about a surprising "Ah Ha!" moment.  "may" ends the sentence mid-clause, requiring a modal verb completion like "may find yourself" or " |
| p0.006 | 31 | 0.342 | 0.658 | Narrative setup completed humorous domestic scenario "you too may" requires completion |
| p0.015 | 29 | 0.401 | 0.599 | TV show description suspenseful narrative "you too, may" |
| p0.03 | 13 | 0.633 | 0.367 | you may |

### Example 64
> _source ctx:_ artropolis art find #1: vadim katznelson
i was completely enthralled by the unique style & colors used by this artist
when i first encounterd this piece…i thought it was plastic
but the curator at the show said he created it
by forcing acrylic resin polymer through cloth ~ cool!
shown at artropolis:

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 130 | 0.648 | 0.352 | Biographical profile format for a photography award winner, following a structured pattern of credentials, achievements, and institutional affiliations.  The sentence ending mid-flow lists "Renee Rappole and Renee Boy" is introducing co-winners (likely co-photographers or co-submitters) with "and,"  |
| p0.0 | 112 | 0.629 | 0.371 | Artist biography following standard museum/catalog format, moving through exhibitions, institutions, and collaborators sequentially.  The text is mid-sentence describing a specific exhibition with collaborators "russell boy" — a truncated proper name (likely "Russell Boyle" or similar photographer)  |
| p0.001 | 89 | 0.734 | 0.266 | Credit/attribution continuation: the text is mid-sentence crediting "Randy Ray & Randy Boy" as collaborators.  Press release format: institutional announcement tone signals formal acknowledgment of production collaborators.  Final token "boy" is part of the collaborator name "Randy Boy" (likely "Ran |
| p0.002 | 64 | 0.582 | 0.418 | Press release format following structured announcement style throughout.  The sentence lists photographer credits, establishing a series pattern requiring completion.  "robert boy" is a truncated name mid-word, strongly constraining next tokens to "d" followed by continuation of the production credi |
| p0.006 | 29 | 1.666 | -0.666 | artist biography format  credits/attribution pattern  "roy roy" likely continues |
| p0.015 | 24 | 0.688 | 0.312 | artist credits production credits "robert boy" |
| p0.03 | 13 | 2.133 | -1.133 | ro boy |

### Example 65
> _source ctx:_ Earlier this year, federal investigators began requesting corporate documents and questioning staff at American Media Inc., the company run by Donald Trump’s longtime friend David Pecker, about a special issue of the National Enquirer it produced that lavished praise on Saudi Arabia and its controve

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 109 | 0.387 | 0.613 | Political/religion news article style, covering Catholic-Protestant tensions, with formal journalistic register throughout.  Narrative momentum building toward explaining what Trump was doing during the papal visit that caused controversy — a direct explanation is expected after "that led to the con |
| p0.0 | 124 | 0.352 | 0.648 | Narrative momentum: the sentence about the "grand tour" is mid-clause, requiring completion of what the tour involved or where it took place.  Thematic pattern: article chronologically traces Obama's foreign policy, so the grand tour detail will likely reference specific destinations like Asia/Afric |
| p0.001 | 89 | 0.365 | 0.635 | News article structure: after background on Trump-Pence friendship, narrative shifts to Trump's political behavior requiring elaboration.  Causal/explanatory momentum: the text establishes a chain explaining Trump's actions; Pence's visit is the latest clue to be developed.  Final token "tour" ends  |
| p0.002 | 75 | 0.448 | 0.552 | Narrative momentum building toward describing Trump's "tour" of foreign capitals, establishing context for the Saudi/Palestine issue.  Chronological political journalism structure moving backward from present event to preceding actions.  Final token "tour" ends an incomplete clause describing Trump' |
| p0.006 | 44 | 0.411 | 0.589 | political investigative journalism  narrative momentum toward Obama's actions  "took him on a goodwill tour" requires completion describing the tour's destination or purpose |
| p0.015 | 25 | 0.528 | 0.472 | political scandal Senate hearings "congressional tour" |
| p0.03 | 13 | 0.971 | 0.029 | tour tour |

### Example 66
> _source ctx:_ About These 3 Apps
Cellular Apps Is Easier Than Ever. Many people have cellphones than ever before. There is much more possibility for the user to get hurt on the street. The possibility is so authentic motorists with diverted attention are caught up in their apparatus and also will need to find inf

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 116 | 0.250 | 0.750 | Listicle/SEO-style blog content pattern: fragmented, loosely connected paragraphs with bullet-point and numbered list structures repeating throughout.  Continuation of "free dating sites for seniors" topic: each paragraph introduces a new angle or feature of online dating for older adults, suggestin |
| p0.0 | 117 | 0.267 | 0.733 | Promotional/marketing register with repetitive structure: each paragraph introduces an online dating platform with benefits and features, establishing a listicle pattern of platform descriptions.  Semantic expectation of continued platform promotion: the text consistently pairs "some other [platform |
| p0.001 | 85 | 0.244 | 0.756 | Continuation of a listicle/FAQ about free dating site alternatives, suggesting more comparative points or examples will follow.  The text contrasts free vs. paid services, implying further elaboration on benefits, drawbacks, or recommendations is expected.  Final sentence ends mid-thought comparing  |
| p0.002 | 79 | 0.237 | 0.763 | Promotional/informational web content about online dating apps, maintaining a structured listicle format with bullet points.  Continuation of app-recommendation section with a pattern of feature highlights and user benefits.  The final sentence ends a paragraph about Facebook and Google+, signaling  |
| p0.006 | 32 | 0.266 | 0.734 | List/enumeration pattern established  Online dating app content  Final sentence introduces additional app types |
| p0.015 | 26 | 0.478 | 0.522 | list continuation advice format "Also you can use..." |
| p0.03 | 11 | 0.376 | 0.624 | _<extraction failed>_ |

### Example 67
> _source ctx:_ CLOVIS, N.M. - Ellen P. Olguin, 67, died Saturday, Oct. 14, 2000, in Lubbock, Texas.
Rosary will be at 7 p.m. today at Our Lady of Guadalupe Catholic Church. Mass will be celebrated at 11 a.m. Wednesday at the church. Burial will be at Texico Cemetery in Texico by Steed-Todd Funeral Home.
Mrs. Olgui

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 160 | — | — | _<extraction failed>_ |
| p0.0 | 117 | 0.612 | 0.388 | Ongoing quote from Sharon Lutz, the author of "The Road to Recovery," mid-sentence requiring completion of "M[ary/Mary Anne]"  The list of community members is being enumerated, establishing a pattern of proper names with titles/roles ("Dale K. Lutz...Mary")  Final token "M" is the beginning of a tr |
| p0.001 | 92 | 0.589 | 0.411 | Obituary/tribute genre with personal anecdotes and community recognition, continuing to list family members.  The phrase "such as M" signals an incomplete list of names, requiring completion of "M" with a given name.  Final token "M" is the beginning of a proper name (likely "Marilyn" or "Marian"),  |
| p0.002 | 68 | 0.748 | 0.252 | Bulleted list of named recipients continuing a consistent naming pattern.  Each recipient follows "Name: Title" format, with a colon after surname.  The final token "M" begins a truncated surname for "Marilyn" completing "Marilyn: [Title]." |
| p0.006 | 32 | 0.911 | 0.089 | Local news narrative continues  charity event details  "M" is the start of a name |
| p0.015 | 26 | 1.025 | -0.025 | obituary format family members listed "MomM" |
| p0.03 | 11 | 1.071 | -0.071 | M |

### Example 68
> _source ctx:_ Garrett County Commissioners oversee day-to-day local decisions on issues such as roads, hiring county employees, allocating funding for special projects, and establishing an annual budget. They oversee several county entities like Economic Development, Community Action, Police and First Responders.

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 117 | 0.427 | 0.573 | Direct quote continuation: a partial quote opened with quotation marks and the word "will" requires completion, almost certainly "best represent" or similar phrasing.  Candidate questionnaire format: the text presents a structured list of questions directed at elected officials, with consistent phra |
| p0.0 | 124 | 0.308 | 0.692 | Political candidate election guide genre, establishing criteria for voter decision-making ("how to choose," "how well they will").  The text is building toward a closing question to voters, asking them to evaluate candidates based on previously listed qualities (fiscal responsibility, economic growt |
| p0.001 | 99 | 0.525 | 0.475 | Direct voter appeal establishing criteria for candidate assessment, requiring completion of the conditional ("candidates who will...").  Parallel rhetorical structure: the text contrasts "will work for me" with "I will be able to make them work for me," building toward candidate qualification.  Fina |
| p0.002 | 64 | 0.321 | 0.679 | Election voter guide format establishing candidate comparison structure.  Logical argument building toward candidate suitability, requiring a concluding assessment.  Final fragment "who do you think will" demands completion with a verb phrase like "best represent your values" or "make your town bett |
| p0.006 | 29 | 0.595 | 0.405 | Voter questionnaire format Candidate evaluation questions "that will" requires completion |
| p0.015 | 26 | 0.450 | 0.550 | election context candidate evaluation "do you believe they will" |
| p0.03 | 13 | 0.861 | 0.139 | will they |

### Example 69
> _source ctx:_ The Veil Nebula is a diffuse nebula located in the northern constellation Cygnus, the Swan. Also known as Witch’s Broom Nebula, Bridal Veil Nebula, Cirrus Nebula, or Filamentary Nebula, it constitutes the visible parts of the Cygnus Loop, a supernova remnant in Cygnus. It is located at an approximat

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 112 | 0.533 | 0.467 | Blog/magazine post about telescope design and optics, establishing domain around DIY telescope building with a specific example lens configuration.  Narrative momentum: the text has introduced the lens design context and is transitioning to the author's practical demonstration, signaling an upcoming |
| p0.0 | 111 | 0.262 | 0.738 | Blog/membership post format: introduction to a website, followed by a membership request form, signaling a call-to-action continuation.  The membership form fields have been listed with labels, and "Website" is being completed mid-entry — a truncated URL or domain name is expected next.  "cele" is t |
| p0.001 | 83 | 0.308 | 0.692 | Blog post footer attribution pattern: author name begins after website description.  Acronym completion required: "cele" is the start of "celestron" or "celestronomer," fitting the astronomy enthusiast context.  Final token "cele" is a truncated word mid-attribution, immediately constraining next to |
| p0.002 | 73 | 0.402 | 0.598 | Copyright/attribution notice pattern: "©Cele" is the beginning of a standard photographer credit line.  Blog post metadata block signals a closing attribution field.  Final token "Cele" is a truncated proper noun (likely "Celestron" or the author's name), requiring immediate continuation of that wor |
| p0.006 | 32 | 0.328 | 0.672 | Website attribution celestial photography site "cele" is "celestron" abbreviation |
| p0.015 | 23 | 0.552 | 0.448 | website attribution astronomy community "cele" |
| p0.03 | 12 | 1.474 | -0.474 | cele |

### Example 70
> _source ctx:_ Promaster Wheel Well Cabinets
Pairs well with...
- DIY Installation
- Policies, Shipping, Warranty
- Installation Services
Elevate your van conversion project to new heights with our Promaster Van Wheel Well Cabinets and create a usable and functional space in the garage of your campervan. Crafted w

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 131 | 0.480 | 0.520 | Product listing page format with structured sections (Product Specs, Dimensions, Ordering, Notes), following retail/boating industry conventions. The "Notes" section signals a customer service/caveat notice.  A list of conditions or constraints for ordering is being enumerated, with "regarding" begi |
| p0.0 | 122 | 0.366 | 0.634 | E-commerce product listing pattern with standard boilerplate sections: description, availability, warranty, and customization notice.  The phrase "changes concerning" is part of a standard legal/corporate disclaimer about custom orders, requiring completion with what the changes apply to — likely "c |
| p0.001 | 75 | 0.328 | 0.672 | E-commerce product page structure: specifications section transitioning to a returns/notes clause.  Legal/disclaimer tone expected: terms conditions language about order modifications or fabric substitutions.  Final token "concerning" opens a relative clause requiring specification of what order con |
| p0.002 | 76 | 0.346 | 0.654 | Product listing details pattern: specifications, materials, and warranty information are being enumerated.  Formal e-commerce product description tone maintained throughout, with legal/disclaimer language.  Final fragment "concerning" is a preposition beginning a list of order customization conditio |
| p0.006 | 35 | 0.426 | 0.574 | Custom woodshop policy statement  Refund/return policy explanation  "concerning" introduces what qualifies |
| p0.015 | 27 | 0.525 | 0.475 | custom wood order retail returns policy "concerning" |
| p0.03 | 13 | 0.935 | 0.065 | regarding |

### Example 71
> _source ctx:_ Choosing a Woolbabe Sleeping Bag is one of the best steps towards introducing a sleep association for your baby, but what is a sleep association and what else can you do to help your Woolbabe bag work it’s magic?\nWhether you call it a sleep association, a sleep prop or routine, we’re talking about 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 139 | 0.278 | 0.722 | Numbered list structure with consistent pattern: each point is a brief tip about dog potty training, implying continuation of the series.  The phrase "and changes" ends the final bullet point, strongly suggesting either a closing clause (e.g., "and changes in routine/diet") completing the thought, o |
| p0.0 | 125 | 0.307 | 0.693 | Continuation of a structured list of situations that trigger separation anxiety in dogs — "big changes" is the final item and more examples are expected (e.g., travel, moving house).  Domain/genre signals: pet care/welfare informational article with a Q&A format, providing practical advice to dog ow |
| p0.001 | 82 | 0.276 | 0.724 | Continuation of a list of stress triggers (vaccination, vet visits, new pets, new humans) requiring completion.  Educational/wellness advice tone maintained throughout, guiding pet owners toward practical, reassuring guidance.  Final phrase "major changes" is mid-list, strongly implying additional e |
| p0.002 | 59 | 0.267 | 0.733 | Product review/article format continues explaining factors affecting puppy sleep patterns.  List-style enumeration of sleep disruptors building toward a comprehensive conclusion.  "and other changes" is mid-list, strongly expecting continuation with specific examples like environments or household r |
| p0.006 | 33 | 0.571 | 0.429 | Pet grooming article context  symptoms/behaviors list  "changes" ends mid-phrase |
| p0.015 | 25 | 0.422 | 0.578 | dog training tips stress causes "and other changes" |
| p0.03 | 14 | 0.786 | 0.214 | and big changes |

### Example 72
> _source ctx:_ Gold Ore Crusher. In gold concentration, high-tech gold mining equipment, such as gold detectors, elegant modern dredgers, and light locks will be needed. The gold crusher was also used as the main crusher in the gold crushing industry. Jaw crusher is the most commonly used mining equipment for gold

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 115 | 0.352 | 0.648 | Search engine results/web scraping content about granite processing equipment, establishing a catalog/product description pattern. The text alternates between equipment listings and descriptive product content.  The pattern shifts into a descriptive sentence about "stone crushing plant," suggesting  |
| p0.0 | 132 | 0.270 | 0.730 | Search results page listing mining equipment and crusher articles, establishing a repetitive pattern of product descriptions, company names, and technical specifications.  The phrase "stone crushing plant is" opens a definitional/technical sentence, following the pattern of the earlier sentence "Cru |
| p0.001 | 96 | 0.286 | 0.714 | Continuation of a structured informational page about stone crushing machinery, maintaining explanatory/educational tone throughout.  Argumentative momentum: "stone crushing plant is" sets up a definitional or descriptive clause elaborating on what it is or does.  Final token "is" opens a predicate  |
| p0.002 | 78 | 0.431 | 0.569 | Structured list of crushing plant FAQs, each with heading followed by explanatory prose.  Continuation of a new FAQ entry beginning with "What is a stone crushing plant is" — establishing definitional content expected.  Final token "is" begins a predicate clause defining what a stone crushing plant  |
| p0.006 | 27 | 0.625 | 0.375 | technical description pattern  product specification context  "is" begins a predicate |
| p0.015 | 25 | 0.335 | 0.665 | equipment description process explanation "stone crushing plant is" |
| p0.03 | 15 | 0.385 | 0.615 | stone crushing plant is |

### Example 73
> _source ctx:_ Painting & Handyman Service Company of Ft Lauderdale, Florida in Broward County
Do you have Painting needs?
Do you need Interior & Exterior Painting?
If you have any painting or Handyman needs, do not hesitate! Call Affordable Painting & Handyman in Ft Lauderdale! We specialize in all types of inter

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 129 | 0.598 | 0.402 | Bulleted list of qualifications/attributes for a lawn care service, each item emphasizing experience and reliability (e.g., "25 years of lawn care," "3+ years").  Marketing/promotional copy pattern established, with benefit-focused language like "reliable," "professional" and "affordable" — next tok |
| p0.0 | 134 | 0.482 | 0.518 | Promotional/marketing copy for a residential lawn care service, establishing credibility and professionalism throughout.  Numerical pattern continuation: "over 3" strongly implies a large number following (e.g., "over 30 years of experience"), a common industry credential phrase in landscaping/servi |
| p0.001 | 90 | 0.510 | 0.490 | Promotional/business description pattern established, emphasizing decades of experience and expertise.  The phrase "over 3" strongly signals a number continuation — completing the age/experience figure, e.g., "35 years of experience."  Final token "3" is the start of an age figure ("over 35 years"), |
| p0.002 | 73 | 0.492 | 0.508 | Service marketing page building credibility through experience and qualifications.  Numerical pattern continuation: "over 3" strongly implies a large number like "30" years or "35 years."  Final token "3" begins a specific numeric year count, constraining next tokens to digits completing a decade fi |
| p0.006 | 28 | 0.661 | 0.339 | Business description pattern  experience credential expected  "3" begins a number |
| p0.015 | 25 | 0.607 | 0.393 | marketing copy experience claims "has over 3" |
| p0.03 | 15 | 0.708 | 0.292 | with over 3 |

### Example 74
> _source ctx:_ Fear of dentistry is an unfortunately common and potentially devastating condition. A surprisingly large number of patients go for years without seeing a dentist because they’re so afraid of undergoing dental treatment. These same patients end up suffering from dental decay, gum disease, and tooth l

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 109 | 0.191 | 0.809 | Promotional/commercial copy for a spa or wellness clinic, using persuasive, benefit-driven language throughout.  List-continuation pattern: "massage therapy, aromatherapy, guided visualization, craniosacral therapy, acupressure, and" signals an incomplete enumerated list requiring more items to comp |
| p0.0 | 119 | 0.214 | 0.786 | Enumerated list structure: "massage therapy," "guided meditation," "acupuncture," "chiropractic adjustments," and — a series of complementary holistic treatments still being listed.  Health/wellness promotional register: the text consistently uses professional, reassuring language describing alterna |
| p0.001 | 81 | 0.208 | 0.792 | Continuation of a list of holistic therapy techniques already introduced (craniosacral therapy, aromatherapy, massage), expecting more examples.  Marketing/promotional register for a wellness practice, maintaining reassuring, inclusive tone throughout.  Final token "and" is a conjunction mid-list, r |
| p0.002 | 78 | 0.228 | 0.772 | Continuation of a list of treatment modalities in a holistic medicine marketing context.  The enumeration pattern ("massage therapy, cranial sacral therapy, and") requires at least one more item to complete the series.  Final token "and" is a list coordinator demanding the immediate next treatment m |
| p0.006 | 29 | 0.313 | 0.687 | wellness treatment list complementary therapies expected "and" continues list |
| p0.015 | 25 | 0.362 | 0.638 | therapeutic methods relaxation techniques "and" |
| p0.03 | 12 | 0.647 | 0.353 | and |

### Example 75
> _source ctx:_ Warwick. Baron Security, a consumer-focused website dealing in matters of security, released its list of the top 50 safest cities, towns and villages in New York State.
The Town of Warwick was listed as the 36th-safest community while the Town of Chester was ranked as the fourth safest.
The Village 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 136 | 0.345 | 0.655 | Rankings list pattern: numbered entries with city names, counties, and ratings (e.g., "86. New Orleans...87...88..."), requiring next rank "147" and its corresponding city/counties to follow.  Structured data format: each entry follows "rank. City, State...County...Rating" pattern, establishing rigi |
| p0.0 | 138 | 0.289 | 0.711 | Structured list pattern: ranked cities with numerical positions (1, 2, 7, 10, 116, 147) followed by city names and rankings, establishing strong continuation expectations.  Ranking list momentum: the enumeration of ranked cities is incomplete; each entry follows "[number]. [City] No. [rank]" format  |
| p0.001 | 118 | 0.390 | 0.610 | Ranking list pattern: ranks 1-147 being listed sequentially with state rankings and scores, each entry following "[rank] [rank position] [score] [state] [score] [state]".  Incomplete sequence reaching rank 147 implies more rankings may follow or the list concludes.  Final token "147" begins a new ra |
| p0.002 | 101 | 0.331 | 0.669 | Structured ranking table with sequential position entries, each following the pattern "[Rank] [Ranking name], [state], [value]"  The list is mid-sequence, enumerating 2014 rankings by rank position  Final token "147" is a rank number mid-entry, requiring an immediate ranking name and state (e.g., "1 |
| p0.006 | 33 | 0.798 | 0.202 | Ranking list continuation State-by-state performance data 147th position requires next entry |
| p0.015 | 28 | 0.809 | 0.191 | ranked list of cities housing affordability index "147" |
| p0.03 | 14 | 0.837 | 0.163 | 147 |

### Example 76
> _source ctx:_ Bill Murray and Ernie Hudson Teach Jimmy How to Wield a Ghostbusters Proton Pack
Bill Murray is an award-winning actor, comedian and writer. He first gained national attention as a cast member on NBC’s Saturday Night Live. After leaving the show in 1980, he successfully transitioned into films. His 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 114 | 0.359 | 0.641 | Narrative momentum: the text is listing cast member details and career highlights, establishing context for a specific announcement or role.  Repetition/continuation pattern: the phrase "The actor will appear in" mirrors earlier mentions of cast roles, strongly predicting a specific film title follo |
| p0.0 | 112 | 0.307 | 0.693 | Celebrity gossip/entertainment news register, listing Scarlett Johansson's upcoming film projects with a pattern of "she will appear in [title]."  The list pattern established with "She will also appear in" signals at least one more movie title is expected to complete the enumeration.  Final token " |
| p0.001 | 93 | 0.336 | 0.664 | News article summarizing multiple cast-related stories about the "Ghostbusters" movie, establishing a pattern of introducing and elaborating on each cast member's activity.  Continuation of list/overview structure: two stories covered, now returning to the main subject for further development.  Fina |
| p0.002 | 64 | 0.350 | 0.650 | Article listing upcoming sequel projects for Ghostbusters 2016 cast members.  Continuation pattern: each cast member gets a brief project description.  Final token "in" begins a prepositional phrase specifying which sequel project Chris Hemsworth appears in next. |
| p0.006 | 32 | 0.441 | 0.559 | news article format continues  casting details expected  "recently in" requires a project name |
| p0.015 | 31 | 0.308 | 0.692 | movie sequel news Evan Almighty context "set to reprise his role in" |
| p0.03 | 14 | 0.401 | 0.599 | will star in |

### Example 77
> _source ctx:_ JOHN JAMES BLUNT (1794-1855), English divine, was born at Newcastle-under-Lyme in Staffordshire, and educated at St John's College, Cambridge, where he took his degree as fifteenth wrangler and obtained a fellowship (1816).
Damian didn't care; Sofia liked Pierre, and he had a feeling Pierre's blunt 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 159 | 0.467 | 0.533 | Scientific anatomical description of *Chelone* (a type of fossil nautilid), requiring precise terminology about shell structure — "fins" and "tubercles" suggest continued anatomical labeling.  The text is mid-sentence describing a structural feature of the shell's internal anatomy, expecting continu |
| p0.0 | 142 | 0.453 | 0.547 | Scientific anatomical description pattern: formal taxonomic/natural history prose detailing morphological features of a mollusk species.  Continuation of an anatomical feature list: the text is mid-sentence describing the radula's morphology, with "a series of teeth on the outer side of the radular  |
| p0.001 | 101 | 0.677 | 0.323 | Comparative anatomical description of chimaeroid fins mid-sentence, requiring continuation of anatomical detail.  Structural contrast pattern: "hard edges on one side... softer ones on" implies a paired anatomical comparison is being completed.  Final phrase "softer ones on" is mid-construction, req |
| p0.002 | 89 | 0.499 | 0.501 | Scientific description of fish anatomy, building detailed morphological account of *Pterolepis*  Structural continuation of a paired anatomical feature: "paired keels on one side and a keel on the other"  Final phrase "a keel on" is mid-sentence, requiring anatomical location completion, likely "the |
| p0.006 | 32 | 0.606 | 0.394 | descriptive biological text anatomical comparison structure "on" requires anatomical noun |
| p0.015 | 26 | 0.527 | 0.473 | scientific description anatomical structures "small teeth on" |
| p0.03 | 14 | 0.934 | 0.066 | the base on |

### Example 78
> _source ctx:_ Cat Tesla's artwork includes both ethereal landscapes and abstract designs. and she is intrigued by texture, color, and the juxtaposition of shapes. Tesla's painting's subjects are organic, either originating from Mother Nature, or inspired by her. One of her goals with her work is to transport the 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 129 | 0.467 | 0.533 | Reflective/motivational writing tone — informal but profound, blending personal gratitude with broader life philosophy about nature and human purpose.  The quote is being completed mid-sentence, echoing the earlier "magic, wonder and beauty" motif; it builds toward a concluding, aspirational stateme |
| p0.0 | 126 | 0.485 | 0.515 | Poetic/reflective prose rhythm: flowing, elevated language ("sigh," "tremble," "whispers," "breathtaking") guides continuation of lyrical tone.  Philosophical meditation on nature and existence: the text contrasts urban alienation with natural wonder, building toward a concluding transcendental stat |
| p0.001 | 92 | 0.429 | 0.571 | Reflective, poetic prose about nature's beauty and presence, establishing philosophical tone throughout.  The quoted phrase "see the artistry" is mid-sentence, requiring completion with a noun or object — likely "in every leaf," "of nature," or similar.  Final token "artistry" ends an incomplete dir |
| p0.002 | 74 | 0.543 | 0.457 | Poetic/reflective prose style maintained throughout, blending nature with spiritual wonder.  Narrative momentum: the passage builds toward appreciating nature's artistry as evidence of something greater.  Final token "artistry" is mid-sentence, requiring a closing quotation or continuation describin |
| p0.006 | 33 | 0.580 | 0.420 | poetic nature reflection tone  celebration of natural beauty  "artistry" requires completion |
| p0.015 | 24 | 0.763 | 0.237 | reflective tone nature writing "artistry" |
| p0.03 | 17 | 1.031 | -0.031 | appreciate the artistry |

### Example 79
> _source ctx:_ Sunday, 28 April 2013
Leviathan Rising: Redux
Mssr Blease has commented on this book here before but I add my own commentary, some years later it must be admitted, as I am thoroughly enjoying this great series.
Leviathan Rising is a wonderful combination of the 20,000 Leagues under the Sea, Murder o

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 133 | 0.286 | 0.714 | The author is building toward completing the sentence with an enthusiastic, anticipatory clause expressing eagerness — "eagerly" signals an immediate action phrase is expected, like "await the publication," "look forward to," or "wait for the release of."  The narrative follows a consistent enthusia |
| p0.0 | 132 | 0.234 | 0.766 | Narrative momentum of anticipation/enthusiasm: the text has built up excitement about the books, concluding with "eagerly" demanding continuation of eagerness-themed actions.  First-person blog voice ("I"), informal yet enthusiastic tone ("can't wait," "dare not say") throughout, guiding continuatio |
| p0.001 | 88 | 0.292 | 0.708 | Enthusiastic book review momentum building toward a concluding anticipation statement. The narrative describes reader eagerness for the sequel.  List/description pattern established: books are being praised, details given, now transition to reader reaction.  Final token "eagerly" opens a participial |
| p0.002 | 73 | 0.278 | 0.722 | Narrative momentum building toward the author's eagerness to read the upcoming series.  The article has followed a pitch structure: setup, characters, series premise, now wrapping up.  "eagerly" ends mid-clause, requiring completion with an infinitive verb like "await" or "prepare." |
| p0.006 | 24 | 0.797 | 0.203 | book review narrative anticipation building and eagerly |
| p0.015 | 25 | 0.485 | 0.515 | book review excitement "eagerly" |
| p0.03 | 13 | 0.539 | 0.461 | and eagerly |

### Example 80
> _source ctx:_ Making a Statement: The Benefits of Colored Concrete Sidewalks for Your Texas Property
When it comes to enhancing the curb appeal of your Texas property, colored concrete sidewalks can make a bold statement. Not only do they provide a durable and long-lasting surface for pedestrians, but they also a

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 125 | 0.705 | 0.295 | FAQ list structure: a Q&A format is established, with each question followed by a brief explanatory answer, continuing the pattern of numbered/subscripted FAQs seen in the intro.  Domain/genre signals: construction/materials industry marketing copy explaining concrete products, maintaining informati |
| p0.0 | 138 | 0.397 | 0.603 | FAQ/Q&A format established throughout: alternating bold questions with detailed answers, predicting continued parallel structure with specific elaboration.  The text is addressing "cost factors" in concrete resurfacing — specific cost ranges, material types, and influencing variables are expected ne |
| p0.001 | 83 | 0.599 | 0.401 | FAQ/Q&A format established, with repeated "How?" questions answered sequentially, so a concluding response is expected.  The answer to the final question about color variety must now follow, contrasting with the previous point about limited options.  Final token ends mid-phrase "concrete," requiring |
| p0.002 | 70 | 0.472 | 0.528 | FAQ format with consistent question-answer structure signals next tokens continue answering.  The text is mid-answer listing pros/cons, continuing the pattern of balanced elaboration.  Final token ends mid-sentence after "concrete" — an incomplete question requires a period or continuation before th |
| p0.006 | 34 | 0.476 | 0.524 | FAQ Q&A format  completing the answer about cost  "concrete" ends the question |
| p0.015 | 29 | 0.653 | 0.347 | FAQ format concrete cost "what is the cost of stamped concrete" |
| p0.03 | 17 | 0.585 | 0.415 | what are the options for concrete |

### Example 81
> _source ctx:_ Sept. 10, 2011
- USA Field Hockey
- Download your Iowa Hawkeye iPhone app!
- Iowa and the Big Ten Network
- Big Ten Network: Free Hawkeye Video
- 24 Hawkeyes to Watch
PROVIDENCE, R.I. — The University of Iowa field hockey team shutout Brown University Saturday, 7-0. The 15th-ranked Hawkeyes move to 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 113 | 0.344 | 0.656 | Sports match recap genre with chronological game narrative, detailing possessions, scoring sequences, and player actions step by step.  Scoring chain momentum: the text is mid-sequence of consecutive scoring possessions, having described a second score before a third goal attempt, requiring completi |
| p0.0 | 128 | 0.244 | 0.756 | Sports game recap format, tracking each player's scoring sequence chronologically throughout the match.  The narrative is building toward a cumulative goal tally for Maura O'Connor — the "third" implies her third goal is about to be described with specific match details (time, circumstance, type of  |
| p0.001 | 88 | 0.250 | 0.750 | Sports game recap narrative following chronological action sequence, detailing scoring moments by player.  List of scoring players and moments building: Biermann's first goal already mentioned, now Kelleher's third goal is in progress.  Final token "third" ends mid-sentence describing Kelleher's thi |
| p0.002 | 77 | 0.349 | 0.651 | Game recap narrative momentum: scoring plays are being detailed sequentially for each team.  Repetition/continuation pattern: each player's goal is described with setup, then score attribution.  Final token "third" is part of "third goal of the game," immediately requiring the word "goal" and likely |
| p0.006 | 31 | 0.575 | 0.425 | sports game recap narrative scoring progression sequence "third" requires a score detail |
| p0.015 | 22 | 0.586 | 0.414 | score recap player highlights "third" |
| p0.03 | 12 | 1.154 | -0.154 | third |

### Example 82
> _source ctx:_ LLANO — A Llano couple accused of bilking their hardware-company employer out of $300,000 worth of merchandise they sold on the Internet have accepted plea deals in the case, officials said.
According to prosecutors, the couple from 2004-2010 sold inventory from the Buttery Co. LLP

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 128 | 0.469 | 0.531 | Legal/regulatory news article following a structured reporting format (who, what, why, impact), requiring continuation of the business/ownership structure description.  Specific legal consequences are being enumerated: "criminal charges, civil complaints, administrative penalties" — the sentence int |
| p0.0 | 111 | 0.476 | 0.524 | Local news article about a local business story, establishing context and narrative momentum toward the legal/political dispute.  The article has introduced both parties and is now detailing the plaintiffs' allegations, so specific grievances or legal claims are expected next.  "The company, B.B. Ro |
| p0.001 | 111 | 0.396 | 0.604 | Legal/crime reporting genre with factual, chronological news article style throughout.  Narrative momentum: case details being established — defendant, charges, and context are being elaborated with "was accused of stealing more than $22,000 from B&B Accounting and Advisory LLP."  Final token "LLP"  |
| p0.002 | 67 | 0.413 | 0.587 | News article narrative building toward legal/financial consequences of the scheme.  Established pattern of presenting facts about the Ponzi scheme and its victims.  Final token "LLP" closes a corporate name mid-sentence, immediately requiring continuation describing the victims' connection to the co |
| p0.006 | 26 | 0.539 | 0.461 | legal dispute narrative  company name continuation  " LLP" closing |
| p0.015 | 24 | 0.592 | 0.408 | legal case summary business partnership "LLP" |
| p0.03 | 160 | — | — | _<extraction failed>_ |

### Example 83
> _source ctx:_ i am beginner in java, i am used to VB.
Here is my case:
1. i've made a desktop application using netbeans
2. it contains main.class that launch the myForm.class.
3. it works when run from its project and netbeans
4. then, i made it became a jar file
5. i put the jar file in a specific folder (eg. a

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 119 | 0.317 | 0.683 | Technical bug report/bug-tracker format with structured sections (title, problem description, logs, user input) establishing formal reporting conventions throughout.  The user is mid-sentence describing what their original code looked like before making the change, requiring continuation of the orig |
| p0.0 | 132 | 0.265 | 0.735 | User is sharing a code snippet (a simple JOptionPane confirmation dialog) and has now posted their "original" code — implying they will show the full original Java code for review/testing purposes.  The question context ("why it doesn't work") expects explanation of why the code isn't functioning, l |
| p0.001 | 80 | 0.247 | 0.753 | Developer seeking assistance with a specific Java Swing GUI problem, expecting code clarification or solution guidance.  Narrative shift: speaker is sharing their own code, implying a code snippet or implementation details follow.  Final token "original" begins a clause describing what they posted — |
| p0.002 | 62 | 0.219 | 0.781 | User providing Java code context and asking for help modifying behavior on button click events.  The narrative is mid-sentence explaining attachment/upload context.  "my original" ends a clause requesting code, strongly implying "code" or "Java file" follows. |
| p0.006 | 30 | 0.339 | 0.661 | code-sharing context explaining modifications "my original" refers to shared code |
| p0.015 | 23 | 0.356 | 0.644 | code explanation original code "my original" |
| p0.03 | 13 | 0.602 | 0.398 | my original |

### Example 84
> _source ctx:_ As Calgary gets ready to connect to the South West Ring
Road, ENMAX is preparing its connections as well.
be resuming work on the distribution system from Anderson Road to 146 Avenue
SW. Some construction will start this month with the majority of the work
happening between January to August 2017. T

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 143 | 1.110 | -0.110 | Structured list continuation: each bullet describes a specific route with numbered leg, direction, street names, and distances — this pattern repeats throughout.  The bullet point is mid-sentence, listing a numbered road segment ("leg 3") with a specific street name beginning "49th Ave," requiring c |
| p0.0 | 128 | 1.192 | -0.192 | Structured address/listing format: the document presents an official address with street, city, state, zip, and directional navigation instructions.  Repetition/continuation pattern: directional navigation text has been listing street intersections and routes, establishing a consistent "turn on X st |
| p0.001 | 98 | 1.038 | -0.038 | Geographic event listing pattern: multiple road closures being enumerated sequentially with street names and dates.  Structured event details: location (San Diego, SD) establishes format of specific streets and dates for each closure.  Final token "62nd St & 40th St" is mid-sentence within a list of |
| p0.002 | 74 | 1.187 | -0.187 | Geographic enumeration pattern listing specific road names in the area.  Address/address details being compiled for multiple locations.  Final token ends mid-phrase "route 32 and route 42 connecting to glacier" — next tokens complete the street name with "glacier [direction] street" or similar suffi |
| p0.006 | 41 | 1.047 | -0.047 | Geographic event listing format  Road infrastructure details pattern  "64th and 66th street and" requires continuation of address |
| p0.015 | 160 | — | — | _<extraction failed>_ |
| p0.03 | 20 | 0.947 | 0.053 | Route 17 and 92 |

### Example 85
> _source ctx:_ |August 23, 2012
It's now been a little more than a month since my procedure. I am completely healed and my hair is growing back. I had such an amazing experience and I talked about every detail with Dr. Bauman during a live TV and radio show taping on August 22nd. Check back here to find out when y

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 122 | 0.495 | 0.505 | Testimonial/review genre with positive sentiment momentum: "I am extremely pleased" sets expectation for a glowing, enthusiastic endorsement about the provider.  Narrative arc of transformation: the review promises "the most dramatic change in my life," requiring elaboration on how the surgery achie |
| p0.0 | 142 | 0.479 | 0.521 | Medical/doctor testimonial format signals a personal endorsement narrative, expecting positive outcome language. The testimonial structure follows patient praise for Dr. Tashjian's rhinoplasty results.  Repetition/continuation pattern: "Dr. Tashjian and one" introduces a paired contrast or completio |
| p0.001 | 86 | 0.544 | 0.456 | Testimonial review format established, requiring completion of personal endorsement of Dr. Laskin.  Personal narrative voice ("I am...I was") momentum building toward describing Dr. Laskin's qualifications and positive experience.  Final token "one" begins a superlative phrase ("one of the few/one w |
| p0.002 | 73 | 0.537 | 0.463 | Patient review/testimonial format establishes personal, grateful tone throughout.  Narrative momentum toward describing Dr. Baskin's qualifications and bedside manner.  Final token "one" begins a superlative phrase "one of the best," requiring continuation with a noun like "of the best surgeons in t |
| p0.006 | 37 | 0.504 | 0.496 | Testimonial review format  Dr. Fazio's credentials described  "and one" requires completion of a paired phrase |
| p0.015 | 27 | 0.678 | 0.322 | testimonial quote Dr. Kao's description "and one" |
| p0.03 | 15 | 0.897 | 0.103 | Dr. and one |

### Example 86
> _source ctx:_ An Israeli journalist is to be indicted for possession of classified Israel Defence Forces (IDF) documents in a decision strongly criticised yesterday by the head of the country's Press Council.
Israel's Attorney General, Yehuda Weinstein, announced yesterday that Uri Blau, a reporter for the libera

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 126 | 0.367 | 0.633 | Investigative journalism article on Israeli military censorship, establishing a pattern of expert testimony with quotes and attributions to support claims.  Quoted speech from Professor Ben-Meir is mid-sentence, completing her argument about censorship's impact on Israeli society; the quote's closin |
| p0.0 | 127 | 0.349 | 0.651 | News article reporting a formal accusation (Knesset vote, government indictment), establishing journalistic tone with formal register throughout.  The article follows an inverted pyramid structure — key facts (veto, charges, Knesset) stated upfront; next expected content elaborates on the indictment |
| p0.001 | 100 | 0.321 | 0.679 | News article reporting on an Israeli Knesset hearing, establishing formal journalistic tone throughout.  Narrative momentum: the article has outlined the committee's concerns and is mid-sentence detailing the Knesset members' stated rationale for their request.  Final token "apparatus" ends an incom |
| p0.002 | 76 | 0.248 | 0.752 | Journalistic investigative article pattern, presenting criticism of Yuli Tamir's committee role.  The argument builds toward exposing limitations in parliamentary oversight of the security apparatus.  Final token "apparatus" is mid-sentence within a direct quote, requiring completion — likely adding |
| p0.006 | 36 | 0.360 | 0.640 | Journalistic report on Knesset investigation  explaining committee's mandate  "the security apparatus" requires completion |
| p0.015 | 27 | 0.442 | 0.558 | journalist report parliamentary inquiry "the security apparatus" |
| p0.03 | 14 | 0.522 | 0.478 | the security apparatus |

### Example 87
> _source ctx:_ Watch: Road workers use Lorry to smooth pavement fix - 'waste of time' says councillor
Video has emerged of road workers using a lorry to apply a smooth finish to their fix on a pot-holed pavement.
The footage was reported by the news organisation, the Armagh i.
Shot from a nearby property on the Ar

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 107 | 0.264 | 0.736 | Official press release genre with formal institutional language, maintaining consistent quote-and-attribution structure throughout.  The Department of Transport spokesperson's statement is mid-sentence, directly responding to the accident inquiry about their policy or safety measures regarding bus d |
| p0.0 | 122 | 0.338 | 0.662 | Formal regulatory/administrative document style, with official body statements and disciplinary findings following a structured press release format throughout.  Narrative momentum: the quote is mid-sentence, presenting the Department's formal response to the disciplinary matter, continuing a patter |
| p0.001 | 94 | 0.246 | 0.754 | Official response format (FAQ structure) signals a formal institutional reply continuing to address concerns about the accident.  The quote is mid-sentence, establishing the Department of Transport's position on road safety standards; a commitment or assurance is expected to follow.  Final token "De |
| p0.002 | 68 | 0.378 | 0.622 | Formal press release/news bulletin style with bureaucratic language from a safety regulator.  Statement continuation expected: the Department's position or response on the incident is being elaborated.  Final token "The Department" begins a new clause continuing the formal regulatory response, immed |
| p0.006 | 32 | 0.434 | 0.566 | Regulatory response format  apology/liability statement structure  "The Department" continues formal statement |
| p0.015 | 23 | 0.480 | 0.520 | official statement regulatory response "The Department" |
| p0.03 | 12 | 0.585 | 0.415 | The Department |

### Example 88
> _source ctx:_ STATEN ISLAND, N.Y. -- Dough By Licastri mourns the loss of its owner and namesake, Stephen Licastri, who passed away suddenly in early June. The Dongan Hills entrepreneur left behind a successful little restaurant -- and a famed dough recipe.
Chef Robert Nilsen will step foot into the position of c

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 148 | 0.430 | 0.570 | Restaurant review genre following structured format: executive summary then detailed critique, now closing with final verdict on dishes. The review has covered menu items in sequence.  The text has been systematically evaluating menu offerings, having addressed specials, main courses, sides, and ent |
| p0.0 | 131 | 0.560 | 0.440 | Ongoing restaurant review structure following a consistent pattern: describe dish, technique, texture, flavor — this pattern strongly predicts continuation of the bahn mi description.  The sentence ending "added to the bahn mi" promises elaboration on what is being added — ingredients or toppings —  |
| p0.001 | 96 | 0.712 | 0.288 | Restaurant review narrative continuing a list of menu innovations and culinary techniques introduced so far.  The review follows a pattern of describing new offerings, alternating between techniques and dishes, suggesting more examples follow.  Final token "bì" completes "banh mi," an incomplete phr |
| p0.002 | 86 | 0.343 | 0.657 | Article profiling Austin's restaurant scene, detailing new openings and trends.  Narrative momentum: description of Bao Brothers' menu items mid-list, with "bánh mì" construction still unresolved.  Final token "bánh mì" is an incomplete noun phrase ("to the bánh mì ___"), requiring a complement like |
| p0.006 | 31 | 0.534 | 0.466 | Restaurant food descriptions menu item elaboration "bánh mì" needs completion |
| p0.015 | 25 | 0.492 | 0.508 | restaurant review menu items "bánh mì" |
| p0.03 | 16 | 0.600 | 0.400 | the bò mì |

### Example 89
> _source ctx:_ This review is the first in a series I intend to write on each of the published works of Gordon H. Clark. That is, I hope to summarize and comment on of each of his many books.
But first we start with Clark’s Ph. D. dissertation. Before he wrote any books for publication he completed his 1929 disser

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 139 | 0.484 | 0.516 | Argumentative thesis establishing a paradox: the work is "well known" yet "inconspicuous," requiring explanation of why this contradiction holds true despite recognition.  The text is building toward a pivotal contrast about the work's accessibility versus scholarly recognition, with the "but" pivot |
| p0.0 | 114 | 0.359 | 0.641 | Academic introduction building toward a revelation about a minor work's significance, establishing contrast between obscurity and value.  The phrase "the treatise, though little-known and seldom-cited, is" sets up a positive evaluative clause asserting the text's importance or scholarly significance |
| p0.001 | 94 | 0.297 | 0.703 | Academic commentary establishing value of a lesser-known scholarly work ("The Origin of Ideas"), building toward its significance.  The phrase "that the treatise is" sets up an evaluative claim about the work's importance, likely "a valuable resource," "often overlooked," or "remarkably insightful." |
| p0.002 | 75 | 0.341 | 0.659 | Academic/forensic critique establishing the work's historical significance and importance.  Argumentative momentum: thesis building that the essay's significance outweighs its flawed status.  Final token "is" opens a predicate clause requiring a description of the essay's importance or value, e.g.,  |
| p0.006 | 30 | 0.437 | 0.563 | academic essay tone  argument about scholarly value  "the text is" requires description |
| p0.015 | 26 | 1.173 | -0.173 | academic reference bibliographical note "this essay is" |
| p0.03 | 14 | 0.661 | 0.339 | the thesis is |

### Example 90
> _source ctx:_ Energy Efficiency in Humanitarian Infrastructure - A Practitioners Guideline
Background of these Guidelines
The findings presented here are based on the desktop and field assessment of the humanitarian infrastructure in Ethiopia’s Gambella region by the consultants. Nevertheless, these findings and 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 122 | 0.322 | 0.678 | Listicle/guide format with enumerated benefits of energy-efficient school buildings, maintaining consistent imperative instructional tone throughout.  Sequential benefit enumeration pattern established: "Save money," "Enhance quality of learning," "Better health and comfort" — the list is clearly on |
| p0.0 | 127 | 0.313 | 0.687 | Bulleted list structure established with four dash-separated items; four items promised, third still in progress, requiring completion and a fourth.  The sentence "An energy" is mid-phrase, beginning the third list item; the word "energy" strongly signals "efficient" or "efficient home" as the expec |
| p0.001 | 89 | 0.341 | 0.659 | Instructional guide structure: text has been giving recommendations, now concluding with a final section on energy efficiency measures.  List/continuation pattern: the document has consistently enumerated strategies (examples, steps, options), suggesting more specific measures follow.  Final token " |
| p0.002 | 80 | 0.268 | 0.732 | List-based document pattern: structured tips with headers followed by detailed sub-sections.  Continuation of a "Guidelines for an energy" header, expecting a policy document completing a standard institutional title.  Final token "energy" is mid-phrase within a new header, immediately requiring a n |
| p0.006 | 31 | 0.327 | 0.673 | practical guide format continues  energy efficiency context established  "an energy" requires completion |
| p0.015 | 25 | 0.367 | 0.633 | energy efficiency topic practical tools "an energy" |
| p0.03 | 13 | 0.586 | 0.414 | an energy |

### Example 91
> _source ctx:_ The Institute has collaborated with S. N. Bose National Centre for Basic Sciences, Kolkata; Anhui University, China, and Universidad de Sevilla, Spain, for the research
Thiruvananthapuram, 10th April 2023: Scientists at the Indian Institute of Science Education and Research (IISER) Thiruvananthapura

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 122 | 0.391 | 0.609 | Structured metadata table pattern: pipe-delimited key-value pairs like "Category\|Education," "Platform\|Android/iOS" establish repeating template rows.  List of contact information details: "Official Website," "Phone No" suggest more fields like Email or Contact Us link may follow in the same table |
| p0.0 | 106 | 0.294 | 0.706 | Contact information list format: a structured phone number is being given, requiring continuation of digits following "9".  Directory/contact page structure: each entry includes name, title, department, and phone, suggesting the next tokens complete this final entry's phone number.  The final token  |
| p0.001 | 87 | 0.297 | 0.703 | Contact information pattern: listing phone numbers and social media handles following a "contact us" section.  Sequential number continuation: the sequence "9" is mid-number, requiring remaining digits to complete the Indian mobile number (10 digits total).  Final token "9" is the start of an incomp |
| p0.002 | 67 | 0.360 | 0.640 | Contact details format: phone number being listed after "9"  Institutional contact page pattern with structured fields (address, website, phone)  The final token "9" is the start of a 10-digit Indian phone number, immediately requiring continuation with more digits |
| p0.006 | 28 | 0.829 | 0.171 | Contact info pattern phone number sequence "9" starts next digit |
| p0.015 | 22 | 0.804 | 0.196 | contact details phone number "9" |
| p0.03 | 12 | 1.694 | -0.694 | 9 |

### Example 92
> _source ctx:_ Youth Sport Health Edition
Often overlooked yet most important component to youth sports participation is the impact on health. Physical and mental health. This week we explore how to make health stays as a top priority.
Health benefits of regular physical activity have been documented for years, ye

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 131 | 0.215 | 0.785 | Motivational/inspirational register established throughout, using sports metaphors to build toward a concluding life-lesson about athletics and character-building.  The essay follows a pattern of "what [something is]—so why it matters," building from specific athletic examples toward universal value |
| p0.0 | 133 | 0.242 | 0.758 | Argumentative momentum: a motivational/advocacy essay building toward a conclusion about character, teamwork, and sportsmanship outweighing competition.  Parallel structure contrast: "more than any" sets up a superlative comparison, contrasting the intangible qualities of character/teamwork against  |
| p0.001 | 89 | 0.265 | 0.735 | Argumentative essay building toward a moral conclusion about youth sports and character development.  Contrastive structure: "any" at the end sets up completing a "not as valuable as..." comparison, requiring a noun phrase like "win" or "medal."  Final token "any" is mid-phrase in a contrasting supe |
| p0.002 | 80 | 0.318 | 0.682 | Argumentative essay building toward a concluding moral about sportsmanship and character over competition.  The "no comparison" rhetorical setup establishes a contrast requiring completion between sportsmanship and athletic achievement.  Final phrase "any" ends an incomplete clause comparing sportsm |
| p0.006 | 32 | 0.623 | 0.377 | sports values argument  contrast between sports and competition  "any" completes "none of any" |
| p0.015 | 22 | 0.616 | 0.384 | sports quote life lessons "any" |
| p0.03 | 12 | 0.828 | 0.172 | any |

### Example 93
> _source ctx:_ Come and paint the day in Coos Bay!
Plein Air Paint Out
Plein Air painters are invited to participate in Coos Art Museum’s “Plein Air Paint Out” adventure which will be taking place early morning through mid-day on Saturday, July 8, 2017. Participating artists will be located around the Coos Bay are

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 120 | 0.566 | 0.434 | Bureaucratic/organizational announcement style with bullet-pointed rules and procedural guidelines, established throughout — formal institutional language expected to continue.  The document systematically lists submission rules in sequence (title requirements, size requirements, entry fee, shipping |
| p0.0 | 123 | 0.341 | 0.659 | Institutional bulletin format with bullet-point sections covering admission rules — the pattern suggests a closing procedural note is being completed.  The phrase "will be retrievable by" sets up a specific group of people or roles (e.g., "original owners," "applicants," "designated persons") who ca |
| p0.001 | 84 | 0.366 | 0.634 | Organizational announcement format establishes procedural detail pattern, guiding toward retrieval/contact instructions.  The sentence structure "will be retrieved by" requires a subject (e.g., "artist" or "the artist") completing the retrieval process.  Final token "by" is a preposition mid-clause, |
| p0.002 | 69 | 0.596 | 0.404 | Bulleted list of drop-off/pick-up rules establishing formal institutional guidance throughout.  The list promises completion of a conditional rule about retrieving items.  Final token "by" opens a temporal/location clause, immediately requiring a time or method phrase like "the end of the school yea |
| p0.006 | 31 | 0.361 | 0.639 | event logistics description  pickup instructions continuation  "could be retrieved by" requires a subject |
| p0.015 | 25 | 0.486 | 0.514 | checkout procedures artist instructions "can be retrieved by" |
| p0.03 | 15 | 0.840 | 0.160 | will be returned by |

### Example 94
> _source ctx:_ The Good Place is an American fantasy-comedy television series created by Michael Schur. The series premiered on September 19, 2016, on NBC.
The series focuses on Eleanor Shellstrop (Kristen Bell), a woman who wakes up in the afterlife and is introduced by Michael (Ted Danson) to "The Good Place", a

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 132 | 0.499 | 0.501 | Narrative setup of an ironic premise: "brought into a world where good people are bad" establishes a dark comedic premise requiring a humorous resolution or twist.  The sentence structure promises completion of an expected phrase: "better person," "more virtuous individual," or similar — the irony d |
| p0.0 | 115 | 0.400 | 0.600 | Narrative momentum: the setup describes a sitcom premise, building toward explaining the protagonist's transformation arc.  Semantic expectation: the phrase "she must become a" strongly signals a role/character category, almost certainly "a better person" or "a morally upright/humane person" — the c |
| p0.001 | 101 | 0.413 | 0.587 | Plot summary establishes a moral transformation arc, requiring completion of the character's redemption.  The phrase "turns out she needs to be a" sets up the ironic goal the protagonist must achieve — likely "better person" or "better human" consistent with the "good person" premise.  Final token " |
| p0.002 | 62 | 0.494 | 0.506 | Narrative setup of irony/reversal: protagonist's good deed backfires on her  Sequential cause-effect structure building toward the comedic twist  "become a" is an incomplete predicate requiring a noun phrase describing the opposite of "bad person" |
| p0.006 | 34 | 0.493 | 0.507 | Narrative summary pattern  moral/character arc development  "become a" requires noun phrase |
| p0.015 | 27 | 0.467 | 0.533 | movie plot summary moral transformation "needs to become a" |
| p0.03 | 14 | 0.645 | 0.355 | become a |

### Example 95
> _source ctx:_ Is the light behind the LCD a candle in the wind for the future of our area?
Micro-LED are not readily available as of yet, but already there is dialogue about whether this new technology will replace LCD and OLED in future. This is due to its strong potential to be a thinner, lighter, and brighter 

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 122 | 1.059 | -0.059 | Ongoing metaphor/analogy pattern: "bulb" has been introduced as a substitute for "light" to emphasize solar lanterns, creating a parallel structure requiring completion.  The article consistently uses accessible analogies (e.g., "as simple as lighting a bulb") to explain technical concepts, making t |
| p0.0 | 133 | 1.093 | -0.093 | Blog post format with structured metadata, author name, and topic tag preceding a teaser — narrative momentum points to completing a metaphorical or ironic expression.  The phrase "as bright as the light" is a well-known idiomatic setup requiring completion, almost certainly "as bright as the light  |
| p0.001 | 85 | 1.169 | -0.169 | Ongoing metaphor about LED lighting being "too bright/light," requiring completion of the idiomatic expression.  The blog post's critical but constructive tone about LED efficiency demands a balanced conclusion acknowledging improvements but still identifying weaknesses.  Final token "as" begins a c |
| p0.002 | 78 | 1.046 | -0.046 | Rhetorical question building toward critique of street lighting efficiency  Continuation of metaphor/analogy comparing traditional bulbs to LED technology  Final token "light" is part of the incomplete phrase "as bright as the light," strongly constraining next word to "as bright as the light bulb/l |
| p0.006 | 35 | 0.960 | 0.040 | analogy explanation in progress metaphor completion expected "as bright as the light" requires continuation |
| p0.015 | 23 | 0.871 | 0.129 | blog post lamp metaphor "the light" |
| p0.03 | 13 | 0.913 | 0.087 | the light |

### Example 96
> _source ctx:_ [isabelle] Defining a recursive function
Dear Isabelle list,
I am defining a datatype and a recursive function as below:
comm = Init "nat list" "nat"
| Cond "comm list"
primrec denoFunDef::"commâMatâMat" where
"denoFunDef (Init m n) p=p"|
"denoFunDef (Cond mcl) p = (case mcl of  â p
| ac#la â(denoFu

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 119 | 0.483 | 0.517 | Question is being resolved by the user seeking help debugging a MATLAB/Optimisation problem — next tokens will explain the issue or ask for clarification.  The narrative "I think the code should be right, but don't know why" sets up a problem statement expecting elaboration, error messages, or a spe |
| p0.0 | 111 | 0.717 | 0.283 | Programming Q&A format (Stack Overflow-style): question raised, explanation given, "So why" signals an unresolved problem requiring explanation of code's issue.  Mathematical physics context: Lagrange multipliers used to enforce angular momentum constraint in a central potential problem, establishin |
| p0.001 | 97 | 0.389 | 0.611 | Code block just closed; author is now explaining confusion about code behavior and seeks clarification.  The phrase "I do not know why" signals a self-posed question requiring elaboration of the specific issue or problem observed.  Final token "why" closes an incomplete explanatory clause — next tok |
| p0.002 | 77 | 0.432 | 0.568 | Question about unexpected MATLAB code output, expecting a diagnostic explanation of why `0` appears instead of expected result.  Domain signals: MATLAB coding error, mathematical logic, matrix manipulation context.  Final token: "why" closes a question about a specific unexpected output, immediately |
| p0.006 | 27 | 0.388 | 0.612 | code debugging question error explanation requested "why" ends question |
| p0.015 | 22 | 0.758 | 0.242 | code error syntax issue "Why is" |
| p0.03 | 15 | 0.562 | 0.438 | don't know why |

### Example 97
> _source ctx:_ Nov 1 2010
IDT Systems, provider of 3D and 2D in-surface decoration solutions, has inked a partnership contract with Ingenia Technology. The agreement will enable IDT to provide Ingenia’s proprietary Laser Surface Authentication (LSA) technology to its consumers.
IDT will incorporate the LSA technol

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 135 | 0.360 | 0.640 | Technical product description pattern: formal press release introducing Tesa's E-Code marking solution, requiring specific technical capabilities and applications.  List/continuation momentum: "full-color marking and in-surface" strongly implies a compound noun phrase completing the paired technolog |
| p0.0 | 125 | 0.307 | 0.693 | Press release/promotional format establishing a formal business announcement tone, with company name, product line, and executive quote already given.  Narrative momentum: the "Our mission is to help clients..." sentence is building toward describing capabilities; the list of services/technologies b |
| p0.001 | 89 | 0.404 | 0.596 | Business/marketing article establishing market opportunity for UV printing in consumer goods and retail.  List continuation pattern: "full-color printing, in-surface" signals a paired series of printing capabilities still being enumerated.  Final token "in-surface" is mid-phrase in a list, requiring |
| p0.002 | 71 | 0.427 | 0.573 | Ongoing list of technologies being described — "in-surface" signals continuation of related terms.  Promotional/marketing tone about product capabilities and business advantages.  The final token "in-surface" is mid-phrase, part of a compound technology description requiring a noun or further modifi |
| p0.006 | 28 | 0.477 | 0.523 | Marketing/product description  technical specification context  "in-surface" requires completion |
| p0.015 | 24 | 0.460 | 0.540 | technical product description display manufacturing "in-surface" |
| p0.03 | 14 | 0.687 | 0.313 | in-surface |

### Example 98
> _source ctx:_ Today 2 Aug Ventura County Mountains
Sunny. Highs from 96 to 104 at low Elevations to the mid 80s to mid 90s at high Elevations. Southwest Winds 10 to 20 Mph in the afternoon.
Tonight 2 Aug Ventura County Mountains
Clear. Lows from the mid 60s to lower 70

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 113 | 0.684 | 0.316 | Weather forecast format established — structured daily summary pattern with temperature ranges and conditions expected.  The sequence has been listing forecast days sequentially (Wednesday → Thursday → Friday), so continuation of Saturday/Friday's temperature range and wind details is expected.  The |
| p0.0 | 140 | 0.478 | 0.522 | Weather forecast format established, following standard meteorological structure (time periods, temperatures, precipitation, winds).  Temperature range completion: "60s and low 70" is mid-sentence, part of a dual temperature zone pattern seen earlier ("high 70s and low 80s"), requiring the number to |
| p0.001 | 91 | 0.533 | 0.467 | Meteorological forecast format with structured temperature/condition details throughout.  Continuation of temperature range pattern: low 60s mentioned, high 70s/80s implied to complete the range.  Final token "70" is mid-number in a temperature range ("mid to upper 70s"), immediately requiring compl |
| p0.002 | 83 | 0.506 | 0.494 | Weather forecast format with structured temperature and condition ranges throughout.  Numerical temperature range being completed: "70s" signals the high bound of the range is coming.  The final token "70" is part of the temperature range "low 60s and high 70[s]," requiring a closing digit and unit  |
| p0.006 | 28 | 0.720 | 0.280 | weather forecast pattern  temperature range completion  "70" requires unit |
| p0.015 | 27 | 0.700 | 0.300 | weather forecast temperature range "60 to 70" |
| p0.03 | 15 | 1.200 | -0.200 | the 70 |

### Example 99
> _source ctx:_ Choosing an Online Casino
If you’re looking for a casino online, there are many different factors to consider. You’ll want to find a site that offers a variety of games, great bonuses, excellent customer service, and secure payment methods. Choosing an online casino that meets all of these requireme

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 125 | 0.249 | 0.751 | Listicle/guide structure with numbered tips: each section follows a heading with explanatory paragraph, establishing expectation of a parallel closing thought.  Semantic momentum: the final sentence begins "Before you sign up for any of these sites, you should" — this is a call-to-action imperative  |
| p0.0 | 127 | 0.279 | 0.721 | Numbered list structure (1-10) with consistent pattern: item number, bold topic, explanatory paragraph. Item 10 is the final entry being completed.  Each entry follows a formula: bold heading + ~3-4 sentences explaining the tip, implications, and caveats — item 10's paragraph is mid-completion.  "yo |
| p0.001 | 99 | 0.201 | 0.799 | Instructional/advisory listicle structure: each sentence offers a tip for casino deposit methods, maintaining consistent advice format.  Final clause "You should" signals an imperative concluding recommendation, likely about verifying legitimacy, comparing sites, or final evaluation steps.  Final to |
| p0.002 | 77 | 0.244 | 0.756 | Structured guide pattern: each section follows "should + verb + explanation" format consistently.  Enumeration of practical tips nearing completion — advice about bonuses, promotions, and payments has been covered.  "you should" opens a final imperative clause, requiring a verb phrase completing the |
| p0.006 | 31 | 0.334 | 0.666 | Guide format with numbered tips Final recommendation expected "you should" opens imperative clause |
| p0.015 | 24 | 0.276 | 0.724 | online casino guide recommendations "you should" |
| p0.03 | 13 | 0.565 | 0.435 | you should |

### Example 100
> _source ctx:_ NEW YORK (DTN) -- New York Mercantile Exchange crude oil futures pared losses after the U.S. Energy Information Administration reported a less-than-expected crude oil stock build as refiners ramped up operations and increased demand last week ahead of this Thanksgiving holiday week.
The EIA reported

| model | tok | NMSE | FVE | explanation |
|---|--:|--:|--:|---|
| base | 120 | 0.272 | 0.728 | Financial journalism covering oil market analysis, maintaining consistent pattern of presenting expert quotes and analyst views before elaborating on specific price predictions.  Narrative momentum builds toward explaining the supply/demand outlook after listing analyst forecasts; the bullish sentim |
| p0.0 | 135 | 0.245 | 0.755 | Financial news report structure: each paragraph introduces a new data point, market insight, or expert statement about crude oil supply/demand conditions.  Narrative momentum toward supply vs. demand analysis: the article consistently weighs supply growth (800k-1.4m bpd) against demand growth, build |
| p0.001 | 82 | 0.248 | 0.752 | Market analysis report tracking crude oil prices, supply/demand metrics, and trade dynamics throughout.  Narrative momentum: after listing supply conditions, the text pivots to a broader market outlook statement requiring elaboration.  Final phrase "is otherwise over-supplied" is mid-sentence, requi |
| p0.002 | 70 | 0.270 | 0.730 | Ongoing analysis of oil market fundamentals, expecting continuation of supply/demand outlook.  Narrative momentum: report building toward market outlook implications after detailing supply factors.  Final token "over supplied" ends a concluding clause mid-thought, immediately expecting elaboration o |
| p0.006 | 32 | 0.348 | 0.652 | Market analysis momentum supply-demand balance discussion "overall over supplied" requires completion describing conditions |
| p0.015 | 25 | 0.376 | 0.624 | market analysis supply/demand dynamics "over supplied" |
| p0.03 | 14 | 0.904 | 0.096 | still over supplied |
