# MMEB Local Candidate Audit

| Dataset | Q | Corpus | Cand med | Cand max | Pos len med | Neg len med | Local uniq text ratio | All same text | Empty text frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FashionIQ | 1000 | 5992 | 1000 | 1000 | 26 | 26 | 0.001 | 1.000 | 0.000 |
| Country211 | 1000 | 211 | 211 | 211 | 8 | 8 | 1.000 | 0.000 | 0.000 |
| CIRR | 1000 | 2197 | 1000 | 1000 | 26 | 26 | 0.001 | 1.000 | 0.000 |
| InfographicsVQA | 1000 | 1730 | 1000 | 1000 | 5 | 8 | 1.000 | 0.000 | 0.000 |
| Visual7W | 1000 | 3480 | 1000 | 1000 | 8 | 13 | 0.997 | 0.000 | 0.000 |
| GQA | 1000 | 6850 | 1000 | 1000 | 31 | 33 | 1.000 | 0.000 | 0.000 |
| ChartQA | 1000 | 1601 | 1000 | 1000 | 4 | 5 | 0.998 | 0.000 | 0.000 |
| A-OKVQA | 1000 | 894 | 894 | 894 | 7 | 7 | 1.000 | 0.000 | 0.000 |
| ScienceQA | 1000 | 736 | 736 | 736 | 10 | 15 | 1.000 | 0.000 | 0.000 |
| OK-VQA | 1000 | 2555 | 1000 | 1000 | 6 | 7 | 1.000 | 0.000 | 0.000 |

## Examples

### FashionIQ
- cand=1000 uniq=1 q=`Find an image to match the fashion image and style note: Is shiny and silver wit...` gt=`Represent the given image.` neg=['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']
- cand=1000 uniq=1 q=`Find an image to match the fashion image and style note: Is grey with black desi...` gt=`Represent the given image.` neg=['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']
- cand=1000 uniq=1 q=`Find an image to match the fashion image and style note: Is a solid red color an...` gt=`Represent the given image.` neg=['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']

### Country211
- cand=211 uniq=211 q=`Identify the country depicted in the image` gt=`Fiji` neg=['Liechtenstein', 'Solomon Islands', 'Sint Maarten', 'Colombia', 'Greenland']
- cand=211 uniq=211 q=`Identify the country depicted in the image` gt=`Maldives` neg=['Fiji', 'Liechtenstein', 'Solomon Islands', 'Sint Maarten', 'Colombia']
- cand=211 uniq=211 q=`Identify the country depicted in the image` gt=`French Guiana` neg=['Fiji', 'Liechtenstein', 'Solomon Islands', 'Sint Maarten', 'Colombia']

### CIRR
- cand=1000 uniq=1 q=`Given an image, find a similar everyday image with the described changes: Show t...` gt=`Represent the given image.` neg=['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']
- cand=1000 uniq=1 q=`Given an image, find a similar everyday image with the described changes: Fewer ...` gt=`Represent the given image.` neg=['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']
- cand=1000 uniq=1 q=`Given an image, find a similar everyday image with the described changes: Two an...` gt=`Represent the given image.` neg=['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']

### InfographicsVQA
- cand=1000 uniq=1000 q=`Represent the given image with the following question: Which social platform has...` gt=`pinterest` neg=['runny nose, cough, sore throat', '167,000', '$59.5b', 'ebola, 224k', '2.5b']
- cand=1000 uniq=1000 q=`Represent the given image with the following question: Which three business type...` gt=`restaurants, interior design, wedding venues` neg=['runny nose, cough, sore throat', 'ebola, 224k', 'plastic', '295', '4146']
- cand=1000 uniq=1000 q=`Represent the given image with the following question: Which two platforms are g...` gt=`linkedin, facebook` neg=['pinterest', 'runny nose, cough, sore throat', '167,000', '2.5b', '4146']

### Visual7W
- cand=1000 uniq=1000 q=`Represent the given image with the following question: What color is the sidewal...` gt=`Gray.` neg=['To receive orders.', 'Plates.', 'Person with camera.', 'A video game controller.', 'Toast.']
- cand=1000 uniq=998 q=`Represent the given image with the following question: Where is the man sitting?` gt=`At the computer.` neg=['To receive orders.', 'Person with camera.', 'A windsurfer.', 'At a stop sign.', 'A tree.']
- cand=1000 uniq=998 q=`Represent the given image with the following question: What is in the photo?` gt=`Food.` neg=['Gray.', 'At a stop sign.', 'Hotel emblem.', 'At the circus.', 'Monks.']

### GQA
- cand=1000 uniq=1000 q=`Represent the given image with the following question: What is this bird called?` gt=`This is a parrot.` neg=['Yes, there is a clock that is gold.', 'Yes, there are both a bed and a picture.', 'No, both the cap and the sign are blue.', 'The girl smiles at the phone.', 'No, the shoes are black and the socks are white.']
- cand=1000 uniq=1000 q=`Represent the given image with the following question: What color is the helmet ...` gt=`The helmet is light blue.` neg=['This is a parrot.', 'The juice is in the top of the image.', 'No, the hat is tan.', 'The trees are green.', 'The vase is in the bottom of the image.']
- cand=1000 uniq=1000 q=`Represent the given image with the following question: Is it an indoors or outdo...` gt=`It is indoors.` neg=['The girl smiles at the phone.', 'The juice is in the top of the image.', 'The plate is square.', 'Yes, the pouch is green.', 'No, there are no spiders or dragons.']

### ChartQA
- cand=1000 uniq=999 q=`Represent the given image with the following question: How many food item is sho...` gt=`14` neg=['11.8', '10878', '[2003, 2004]', 'Chinese', '108']
- cand=1000 uniq=999 q=`Represent the given image with the following question: What is the difference in...` gt=`0.57` neg=['14', '11.8', '10878', '[2003, 2004]', '108']
- cand=1000 uniq=999 q=`Represent the given image with the following question: How many bars are shown i...` gt=`3` neg=['14', '11.8', '[2003, 2004]', '327', '6.99']

### A-OKVQA
- cand=894 uniq=894 q=`Represent the given image with the following question: What is in the motorcycli...` gt=`cigarette` neg=['1000', 'eleven', 'i do', 'malaysia', 'trophy']
- cand=894 uniq=894 q=`Represent the given image with the following question: Which number birthday is ...` gt=`thirty` neg=['cigarette', '1000', 'eleven', 'i do', 'malaysia']
- cand=894 uniq=894 q=`Represent the given image with the following question: What best describes the p...` gt=`dirty` neg=['cigarette', '1000', 'eleven', 'i do', 'malaysia']

### ScienceQA
- cand=736 uniq=736 q=`Represent the given image with the following question: Which of the following co...` gt=`how steady a parachute with a 1 m vent was at 200 km per hour` neg=['climate', 'Bryum moss plants are made up of tiny units called cells.', 'chirping', 'Grenada', 'kelp bass']
- cand=736 uniq=736 q=`Represent the given image with the following question: What is the name of the c...` gt=`New Hampshire` neg=['how steady a parachute with a 1 m vent was at 200 km per hour', 'climate', 'Bryum moss plants are made up of tiny units called cells.', 'chirping', 'Grenada']
- cand=736 uniq=736 q=`Represent the given image with the following question: Which of these organisms ...` gt=`mushroom` neg=['how steady a parachute with a 1 m vent was at 200 km per hour', 'climate', 'Bryum moss plants are made up of tiny units called cells.', 'chirping', 'Grenada']

### OK-VQA
- cand=1000 uniq=1000 q=`Represent the given image with the following question: What sport can you use th...` gt=`race` neg=['bake it', 'dry hand', 'pun', 'philadelphia', 'band']
- cand=1000 uniq=1000 q=`Represent the given image with the following question: Name the type of plant th...` gt=`vine` neg=['bake it', 'dry hand', 'philadelphia', 'delivery', 'free']
- cand=1000 uniq=1000 q=`Represent the given image with the following question: What toy is this?` gt=`stuffed animal` neg=['race', 'dry hand', 'pun', 'french open', 'levis']
