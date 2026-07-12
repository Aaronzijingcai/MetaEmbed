# MMEB Raw Sample Audit
Dataset path: `/MURE-V2/code/MetaEmbed/data_dir/MoCa_train_with_image`

## InfographicsVQA
- loaded_len: 23945
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nRepresent the given image with the following question: Which type of fonts offer better readability in printed works?\n | serif fonts | his, he, hers ; home printed |
| 1 | True | False | <|image_1|>\nRepresent the given image with the following question: Which fonts are suited for the web?\n | sans serif | 1889-1890 ; G I A N T S |
| 2 | True | False | <|image_1|>\nRepresent the given image with the following question: Which medium has the highest resolution?\n | print | Duke ; blogs |
| 10 | True | False | <|image_1|>\nRepresent the given image with the following question: Grotesque is an example of serif or sans?\n | sans | water buffalo ; silicone |
| 99 | True | False | <|image_1|>\nRepresent the given image with the following question: How many different Sushi are available?\n | 5 | 6.8% ; 78 |

## ChartQA
- loaded_len: 28298
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nRepresent the given image with the following question: Is the value of Favorable 38 in 2015?\n | Yes | No ; Unchanged |
| 1 | True | False | <|image_1|>\nRepresent the given image with the following question: How many values are below 40 in Unfavorable graph?\n | 6 | 53.84 ; 9.5 |
| 2 | True | False | <|image_1|>\nRepresent the given image with the following question: In which year the value was 51?\n | 2014 | Senior compliance manager ; 2019-20 |
| 10 | True | False | <|image_1|>\nRepresent the given image with the following question: What's the rightmost value dark brown graph?\n | 47 | 2132200 ; 58 |
| 99 | True | False | <|image_1|>\nRepresent the given image with the following question: Is the sum of all the values in 2017 smaller than sum of all the values in 2005?\n | No | Yes ; I don't know |

## A-OKVQA
- loaded_len: 17055
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nRepresent the given image with the following question: What is the man by the bags awaiting?\n | cab | train ; parking time |
| 1 | True | False | <|image_1|>\nRepresent the given image with the following question: Where does this man eat pizza?\n | office | cafe ; commercial office |
| 2 | True | False | <|image_1|>\nRepresent the given image with the following question: What is the occupation of the person driving?\n | farmer | musician ; utility worker |
| 10 | True | False | <|image_1|>\nRepresent the given image with the following question: How did these frisbee throwers get to this location?\n | bike | motorcycle ; bike trail |
| 99 | True | False | <|image_1|>\nRepresent the given image with the following question: What job does the person with the larger item on their head hold?\n | mascot | janitor ; sponsors show |

## DocVQA
- loaded_len: 39462
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nRepresent the given image with the following question: what is the date mentioned in this letter?\n | 1/8/93 | 398,627.92 ; 5/16/80 |
| 1 | True | False | <|image_1|>\nRepresent the given image with the following question: what is the contact person name mentioned in letter?\n | P. Carter | March 8, 1926 ; KATHERINE COOPER |
| 2 | True | False | <|image_1|>\nRepresent the given image with the following question: Which corporation's letterhead is this?\n | Brown & Williamson Tobacco Corporation | Tentative agenda ; WARNER-LAMBERT COMPANY |
| 10 | True | False | <|image_1|>\nRepresent the given image with the following question: Which part of Virginia is this letter sent from\n | Richmond | 07-Jun-95 ; HILLSBORD BRANCH |
| 99 | True | False | <|image_1|>\nRepresent the given image with the following question: What is the date mentioned?\n | May 29, 1990 | 3:30 PM - 3:45PM ; May 15, 2002 |

## OK-VQA
- loaded_len: 9008
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nRepresent the given image with the following question: What is the hairstyle of the blond called?\n | pony tail | husky ; backhand |
| 1 | True | False | <|image_1|>\nRepresent the given image with the following question: How old do you have to be in canada to do this?\n | 18 | handle ; 1823 |
| 2 | True | False | <|image_1|>\nRepresent the given image with the following question: Can you guess the place where the man is playing?\n | aspen | tourist attraction ; canada |
| 10 | True | False | <|image_1|>\nRepresent the given image with the following question: Which breed of dog it this?\n | doberman | sectional ; skinny |
| 99 | True | False | <|image_1|>\nRepresent the given image with the following question: Which first lady campaigned for healthier school meals?\n | michelle obama | hawaii ; valentine |

## Visual7W
- loaded_len: 50000
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nRepresent the given image with the following question: What is written on the white square on the bus?\n | Fox's Ginger Biscuits. | Mac's Macaroni Hut. ; F/S. |
| 1 | True | False | <|image_1|>\nRepresent the given image with the following question: What kind of bus is this?\n | Double decker bus. | City bus. ; Freightliner. |
| 2 | True | False | <|image_1|>\nRepresent the given image with the following question: When is this scene taking place?\n | Day time. | Morning. ; In summer. |
| 10 | True | False | <|image_1|>\nRepresent the given image with the following question: Where is the man working?\n | In an office or study room of some sort. | On the roof. ; To use with the computers. |
| 99 | True | False | <|image_1|>\nRepresent the given image with the following question: Who is wearing a watch?\n | The lady. | The man. ; The woman on the cell phone. |

## MSCOCO_t2i
- loaded_len: 25000
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, pos_image, neg_image_0, neg_image_1, neg_image_2`
- checked: 5, query_has_image: 0, pos_has_image: 5
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 5
- neg_image_slots_present: `{'neg_image_path': 5, 'neg_image_0': 5, 'neg_image_1': 5, 'neg_image_2': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | False | True | Find me an everyday image that matches the given caption: A teddy bear shop is equipped with a door guard teddy and a neighbor teddy above.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 1 | False | True | Find me an everyday image that matches the given caption: A simply decorated room is shown with a blue couch.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 2 | False | True | Find me an everyday image that matches the given caption: The two woman are walking side by side in the road holding umbrellas to shield themselves from the rain.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 10 | False | True | Find me an everyday image that matches the given caption: Young boy showing happy emotion during baseball game.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 99 | False | True | Find me an everyday image that matches the given caption: A bench is sitting by a forest covered in snow.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |

## MSCOCO_i2t
- loaded_len: 25000
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nFind an image caption describing the given everyday image.\n | A skateboarder in mid air following a jump form cincrete. | A young man rides up a ramp on his skateboard. ; The skateboarder is jumping up in the air. |
| 1 | True | False | <|image_1|>\nFind an image caption describing the given everyday image.\n | A table topped with a vase with two pink roses. | A vase filled with flowers sitting on a table. ; A vase filled with pink flowers on top of a table. |
| 2 | True | False | <|image_1|>\nFind an image caption describing the given everyday image.\n | Several pillows are lined up down the length of a bed. | A bed sitting in a attic next to a  table with a lamp. ; A bedroom with a bed, lamps and a balcony. |
| 10 | True | False | <|image_1|>\nFind an image caption describing the given everyday image.\n | There are two luggage cases sitting behind two backpacks. | A line of luggage piled on top of each other. ; Four suitcases stacked on top of each other. |
| 99 | True | False | <|image_1|>\nFind an image caption describing the given everyday image.\n | A person with a umbrella on a city street. | A person walks underneath an open umbrella. ; A woman holding a pink umbrella walking in the rain. |

## VisualNews_t2i
- loaded_len: 25000
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, pos_image, neg_image_0, neg_image_1, neg_image_2`
- checked: 5, query_has_image: 0, pos_has_image: 5
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 5
- neg_image_slots_present: `{'neg_image_path': 5, 'neg_image_0': 5, 'neg_image_1': 5, 'neg_image_2': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | False | True | Retrieve an image of this news caption. Turkish President Tayyip Erdogan looks on after arriving at Esenboga Airport in Ankara Turkey June 8 2015 Turkey faced the prospect of weeks of political turmoil after the ruling AK Party lost its par... | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 1 | False | True | Retrieve an image of this news caption. Ryan Crouser eyes the shot as he prepared to throw Friday at the US Olympic Track and Field Trials in Eugene Ore Crouser won the event with a personal best of 72 feet 6 12 inches.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 2 | False | True | Retrieve an image of this news caption. Cucidati.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 10 | False | True | Retrieve an image of this news caption. Jersey Shore cast.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 99 | False | True | Retrieve an image of this news caption. A bullet cartridge under a microscope in the firearmsanalysis section of the Virginia Department of Forensic Science laboratory in Manassas.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |

## VisualNews_i2t
- loaded_len: 25000
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image`
- checked: 5, query_has_image: 5, pos_has_image: 0
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | False | <|image_1|>\nFind a caption for the news in the given photo.\n | Turkish President Tayyip Erdogan looks on after arriving at Esenboga Airport in Ankara Turkey June 8 2015 Turkey faced the prospect of weeks of political turmoil after the ruling AK Party lost its parliamentary majority REUTERSUmit Bektas. | Defence Minister Panos Kammenos likened the situation to a coup by foreign leaders. ; There has been a stream of demonstrations against the bombings in Ankara throughout the past week. |
| 1 | True | False | <|image_1|>\nFind a caption for the news in the given photo.\n | Ryan Crouser eyes the shot as he prepared to throw Friday at the US Olympic Track and Field Trials in Eugene Ore Crouser won the event with a personal best of 72 feet 6 12 inches. | I also work as a pallbearer it might provide inspiration for another play. ; Geek to champ Sam Priestley took part in a yearlong experiment to become a top table tennis player. |
| 2 | True | False | <|image_1|>\nFind a caption for the news in the given photo.\n | Cucidati. | A St Louis institution World s Fair Donuts have been serving perfection via baked goods for over 30 years. ; Friggen Fried Ice Cream. |
| 10 | True | False | <|image_1|>\nFind a caption for the news in the given photo.\n | Jersey Shore cast. | A path AP photo. ; OphQi. |
| 99 | True | False | <|image_1|>\nFind a caption for the news in the given photo.\n | A bullet cartridge under a microscope in the firearmsanalysis section of the Virginia Department of Forensic Science laboratory in Manassas. | AP PhotoCharlie Riedel. ; The UH60 helicopter like this one was on a training exercise. |

## VisDial
- loaded_len: 12500
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, pos_image, neg_image_0, neg_image_1, neg_image_2`
- checked: 5, query_has_image: 0, pos_has_image: 5
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 5
- neg_image_slots_present: `{'neg_image_path': 5, 'neg_image_0': 5, 'neg_image_1': 5, 'neg_image_2': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | False | True | Represent the given dialogue about an image, which is used for image retrieval: Q:is this a child or adult\nA:adult\nQ:male or female\nA:male\nQ:are they inside or outside\nA:inside\nQ:are they laying on the floor\nA:yes, but there is a bla... | <|image_1|>\nRepresent the given image\n |  ; <|image_1|>\nRepresent the given image\n |
| 1 | False | True | Represent the given dialogue about an image, which is used for image retrieval: Q:what color is horse\nA:brown, but it's black and white photo\nQ:is this outdoors\nA:yes\nQ:do you see any horses\nA:yes, 1\nQ:how about fences\nA:no\nQ:do you... | <|image_1|>\nRepresent the given image\n |  ; <|image_1|>\nRepresent the given image\n |
| 2 | False | True | Represent the given dialogue about an image, which is used for image retrieval: Q:how many bikes there\nA:3\nQ:what color are bikes\nA:i see green red and white\nQ:are they parked on stock parking\nA:no\nQ:are there any people\nA:2\nQ:what ... | <|image_1|>\nRepresent the given image\n |  ; <|image_1|>\nRepresent the given image\n |
| 10 | False | True | Represent the given dialogue about an image, which is used for image retrieval: Q:is oven hot\nA:yes\nQ:do u see red flames\nA:no\nQ:what else is on pizza\nA:olives\nQ:does the pizza look tasty\nA:yes\nQ:is pizza cut\nA:no\nQ:what color is ... | <|image_1|>\nRepresent the given image\n |  ; <|image_1|>\nRepresent the given image\n |
| 99 | False | True | Represent the given dialogue about an image, which is used for image retrieval: Q:what shape is the mirror\nA:rectangle\nQ:is there anything else on the wall\nA:yes\nQ:what else is on the wall\nA:television\nQ:is it a flat screen television... | <|image_1|>\nRepresent the given image\n |  ; <|image_1|>\nRepresent the given image\n |

## WebQA
- loaded_len: 12500
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, pos_image, neg_image_0, neg_image_1, neg_image_2`
- checked: 5, query_has_image: 0, pos_has_image: 5
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 0
- neg_image_slots_present: `{'neg_image_path': 5, 'neg_image_0': 5, 'neg_image_1': 5, 'neg_image_2': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | False | True | <|image_1|>\nFind a Wikipedia image that answers this question: Which body part is found on both a 1913 D Barber half and a 1914 Barber Quarter?\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: 1913-D Barber half obverse.\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: NNC-US-1854-G$3-Indian Princess Head 185... ; <|image_1|>\nRepresent the given Wikipedia image with related text information: NNC-US-1907-G$20-Saint Gaudens (Arabic) ... |
| 1 | False | True | <|image_1|>\nFind a Wikipedia image that answers this question: Is more of the building on the corner of King William St-Gracechurch St green or grey?\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: King William St-Gracechurch St.\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: TKMaxxLondon1 TK Maxx, Gracechurch Stree... ; <|image_1|>\nRepresent the given Wikipedia image with related text information: Royal College of Organists, former headq... |
| 2 | False | True | <|image_1|>\nFind a Wikipedia image that answers this question: Do staff at both Metropolitan Transport Authority wear masks at all times during meeting, cleaning and distribution?\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: MTA Leadership Directs Coronavirus Response (49666340833).\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: Grand Central Terminal, NY - panoramio.\... ; <|image_1|>\nRepresent the given Wikipedia image with related text information: NY Katz.\n |
| 10 | False | True | <|image_1|>\nFind a Wikipedia image that answers this question: Is the painted flag of Argentina at the Antarctic settlement "Esperanza" smaller than the flags on top of the German Antarctic research base Neumayer Station III?\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: ArgentineAntarcticEsperanza  Argentine Antarctic settlement "Esperanza" seen from onboard ship in Hope Bay.\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: Central part of the eastern front, Neues... ; <|image_1|>\nRepresent the given Wikipedia image with related text information: Youth Olympic Village, Buenos Aires 2018... |
| 99 | False | True | <|image_1|>\nFind a Wikipedia image that answers this question: What kind of animal is close to a human in both Portrait of a Young Nobleman and the painting by Georges Seurat?\n | <|image_1|>\nRepresent the given Wikipedia image with related text information: Portrait of a Young Nobleman by Nicolas de Largilliere, c. 1714, oil on canvas - National Museum of Western Art, Tokyo - DSC08590  Painting in the National Muse... | <|image_1|>\nRepresent the given Wikipedia image with related text information: Paul Gosselin Self portrait of the Belgi... ; <|image_1|>\nRepresent the given Wikipedia image with related text information: Claude monet jeune fille dans le jardin ... |

## CIRR
- loaded_len: 26115
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image, pos_image, neg_image_0, neg_image_1, neg_image_2`
- checked: 5, query_has_image: 5, pos_has_image: 5
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 5
- neg_image_slots_present: `{'neg_image_path': 5, 'neg_image_0': 5, 'neg_image_1': 5, 'neg_image_2': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | True | <|image_1|>\nGiven an image, find a similar everyday image with the described changes: Many hamster together playing in a different background.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 1 | True | True | <|image_1|>\nGiven an image, find a similar everyday image with the described changes: Sliced oranges in a white table.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 2 | True | True | <|image_1|>\nGiven an image, find a similar everyday image with the described changes: Unlike the reference image, the target image shows two snowplows with orange plow blades.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 10 | True | True | <|image_1|>\nGiven an image, find a similar everyday image with the described changes: Show a border collie running on some grass.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 99 | True | True | <|image_1|>\nGiven an image, find a similar everyday image with the described changes: The target photo has a dark gray train on tracks.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |

## NIGHTS
- loaded_len: 15940
- features: `qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, qry_image, pos_image, neg_image_0, neg_image_1, neg_image_2`
- checked: 5, query_has_image: 5, pos_has_image: 5
- empty_qry: 0, empty_pos_text: 0, empty_neg_text: 0
- pos_equals_query_text: 0, neg_text_contains_pos_text: 5
- neg_image_slots_present: `{'neg_image_path': 5, 'neg_image_0': 5, 'neg_image_1': 5, 'neg_image_2': 5}`

| idx | q_img | p_img | qry | pos_text | neg_text preview |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | True | True | <|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 1 | True | True | <|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 2 | True | True | <|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 10 | True | True | <|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
| 99 | True | True | <|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n | <|image_1|>\nRepresent the given image.\n | <|image_1|>\nRepresent the given image.\n ; <|image_1|>\nRepresent the given image.\n |
