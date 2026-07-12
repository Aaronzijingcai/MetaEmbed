# MMEB Eval BEIR Audit

| Dataset | Q | Corpus | Q img/200 | C img/200 | Q len med | C len med | Corpus uniq | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FashionIQ | 1000 | 5992 | 200 | 200 | 117.0 | 27.0 | 0.000 | many duplicate labels; image-query -> image corpus |
| Country211 | 1000 | 211 | 200 | 0 | 43.0 | 8.0 | 1.000 | short corpus text; image/query -> text corpus |
| CIRR | 1000 | 2197 | 200 | 200 | 125.0 | 27.0 | 0.000 | many duplicate labels; image-query -> image corpus |
| InfographicsVQA | 1000 | 1730 | 200 | 0 | 116.0 | 8.0 | 1.000 | short corpus text; image/query -> text corpus |
| Visual7W | 1000 | 3480 | 200 | 0 | 82.0 | 13.0 | 1.000 | short corpus text; image/query -> text corpus |
| GQA | 1000 | 6850 | 200 | 0 | 95.0 | 33.0 | 1.000 | image/query -> text corpus |
| ChartQA | 1000 | 1601 | 200 | 0 | 109.0 | 5.0 | 1.000 | short corpus text; image/query -> text corpus |
| A-OKVQA | 1000 | 894 | 200 | 0 | 97.0 | 7.0 | 1.000 | short corpus text; image/query -> text corpus |
| ScienceQA | 1000 | 736 | 200 | 0 | 106.0 | 15.0 | 1.000 | short corpus text; image/query -> text corpus |
| OK-VQA | 1000 | 2555 | 200 | 0 | 96.0 | 7.0 | 1.000 | short corpus text; image/query -> text corpus |
| DocVQA | 1000 | 4463 | 200 | 0 | 101.0 | 11.0 | 1.000 | short corpus text; image/query -> text corpus |
| TextVQA | 1000 | 3343 | 200 | 0 | 89.0 | 8.0 | 1.000 | short corpus text; image/query -> text corpus |
| VizWiz | 1000 | 1436 | 200 | 0 | 81.0 | 11.0 | 1.000 | short corpus text; image/query -> text corpus |
| ImageNet-1K | 1000 | 1000 | 200 | 0 | 45.0 | 16.0 | 1.000 | short corpus text; image/query -> text corpus |
| SUN397 | 1000 | 397 | 200 | 0 | 38.0 | 11.0 | 1.000 | short corpus text; image/query -> text corpus |
| VOC2007 | 1000 | 20 | 200 | 0 | 39.0 | 5.0 | 1.000 | short corpus text; image/query -> text corpus |
| ImageNet-A | 1000 | 1000 | 200 | 0 | 45.0 | 16.0 | 1.000 | short corpus text; image/query -> text corpus |
| ImageNet-R | 1000 | 200 | 200 | 0 | 45.0 | 8.0 | 1.000 | short corpus text; image/query -> text corpus |
| ObjectNet | 1000 | 113 | 200 | 0 | 39.0 | 8.0 | 1.000 | short corpus text; image/query -> text corpus |

## Samples

### FashionIQ
- query samples: `['Find an image to match the fashion image and style note: Is shiny and silver with shorter sleeves and fit and flare.', 'Find an image to match the fashion image and style note: Is grey with black design and is a light printed short dress.', 'Find an image to match the fashion image and style note: Is a solid red color and shorter and tighter with more blue and...']`
- corpus samples: `['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']`
- top corpus texts: `[('Represent the given image.\n', 5992)]`

### Country211
- query samples: `['Identify the country depicted in the image', 'Identify the country depicted in the image', 'Identify the country depicted in the image']`
- corpus samples: `['Fiji', 'Liechtenstein', 'Solomon Islands', 'Sint Maarten', 'Colombia']`
- top corpus texts: `[('Fiji', 1), ('Liechtenstein', 1), ('Solomon Islands', 1), ('Sint Maarten', 1), ('Colombia', 1)]`

### CIRR
- query samples: `['Given an image, find a similar everyday image with the described changes: Show three bottles of soft drink.', 'Given an image, find a similar everyday image with the described changes: Fewer paper towels per pack.', 'Given an image, find a similar everyday image with the described changes: Two animals that are of a different species fr...']`
- corpus samples: `['Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.', 'Represent the given image.']`
- top corpus texts: `[('Represent the given image.\n', 2197)]`

### InfographicsVQA
- query samples: `['Represent the given image with the following question: Which social platform has heavy female audience?', 'Represent the given image with the following question: Which three business types is Pinterest good for?', 'Represent the given image with the following question: Which two platforms are good for B2B companies?']`
- corpus samples: `['pinterest', 'runny nose, cough, sore throat', '167,000', '$59.5b', 'ebola, 224k']`
- top corpus texts: `[('pinterest', 1), ('runny nose, cough, sore throat', 1), ('167,000', 1), ('$59.5b', 1), ('ebola, 224k', 1)]`

### Visual7W
- query samples: `['Represent the given image with the following question: What color is the sidewalk?', 'Represent the given image with the following question: Where is the man sitting?', 'Represent the given image with the following question: What is in the photo?']`
- corpus samples: `['Gray.', 'To receive orders.', 'Plates.', 'Person with camera.', 'A video game controller.']`
- top corpus texts: `[('Gray.', 1), ('To receive orders.', 1), ('Plates.', 1), ('Person with camera.', 1), ('A video game controller.', 1)]`

### GQA
- query samples: `['Represent the given image with the following question: What is this bird called?', 'Represent the given image with the following question: What color is the helmet in the middle of the image?', 'Represent the given image with the following question: Is it an indoors or outdoors scene?']`
- corpus samples: `['This is a parrot.', 'Yes, there is a clock that is gold.', 'Yes, there are both a bed and a picture.', 'No, both the cap and the sign are blue.', 'The girl smiles at the phone.']`
- top corpus texts: `[('This is a parrot.', 1), ('Yes, there is a clock that is gold.', 1), ('Yes, there are both a bed and a picture.', 1), ('No, both the cap and the sign are blue.', 1), ('The girl smiles at the phone.', 1)]`

### ChartQA
- query samples: `['Represent the given image with the following question: How many food item is shown in the bar graph?', 'Represent the given image with the following question: What is the difference in value between Lamb and Corn?', 'Represent the given image with the following question: How many bars are shown in the chart?']`
- corpus samples: `['14', '11.8', '10878', '[2003, 2004]', 'Chinese']`
- top corpus texts: `[('14', 1), ('11.8', 1), ('10878', 1), ('[2003, 2004]', 1), ('Chinese', 1)]`

### A-OKVQA
- query samples: `["Represent the given image with the following question: What is in the motorcyclist's mouth?", 'Represent the given image with the following question: Which number birthday is probably being celebrated?', 'Represent the given image with the following question: What best describes the pool of water?']`
- corpus samples: `['cigarette', '1000', 'eleven', 'i do', 'malaysia']`
- top corpus texts: `[('cigarette', 1), ('1000', 1), ('eleven', 1), ('i do', 1), ('malaysia', 1)]`

### ScienceQA
- query samples: `["Represent the given image with the following question: Which of the following could Gordon's test show?", 'Represent the given image with the following question: What is the name of the colony shown?', 'Represent the given image with the following question: Which of these organisms contains matter that was once part of th...']`
- corpus samples: `['how steady a parachute with a 1 m vent was at 200 km per hour', 'climate', 'Bryum moss plants are made up of tiny units called cells.', 'chirping', 'Grenada']`
- top corpus texts: `[('how steady a parachute with a 1 m vent was at 200 km per hour', 1), ('climate', 1), ('Bryum moss plants are made up of tiny units called cells.', 1), ('chirping', 1), ('Grenada', 1)]`

### OK-VQA
- query samples: `['Represent the given image with the following question: What sport can you use this for?', 'Represent the given image with the following question: Name the type of plant this is?', 'Represent the given image with the following question: What toy is this?']`
- corpus samples: `['race', 'bake it', 'dry hand', 'pun', 'philadelphia']`
- top corpus texts: `[('race', 1), ('bake it', 1), ('dry hand', 1), ('pun', 1), ('philadelphia', 1)]`

### DocVQA
- query samples: `['Represent the given image with the following question: What is the ‘actual’ value per 1000, during the year 1975?', 'Represent the given image with the following question: What is name of university?', 'Represent the given image with the following question: What is the name of the company?']`
- corpus samples: `['0.28', '1992', 'PROFESSOR', 'Mr. Heeley', '18.40']`
- top corpus texts: `[('0.28', 1), ('1992', 1), ('PROFESSOR', 1), ('Mr. Heeley', 1), ('18.40', 1)]`

### TextVQA
- query samples: `['Represent the given image with the following question: what is the brand of this camera?', 'Represent the given image with the following question: what does the small white text spell?', 'Represent the given image with the following question: what kind of beer is this?']`
- corpus samples: `['dakota', 'rose', 'cognac', 'bevi', '12:39']`
- top corpus texts: `[('dakota', 1), ('rose', 1), ('cognac', 1), ('bevi', 1), ('12:39', 1)]`

### VizWiz
- query samples: `['Represent the given image with the following question: Can you tell me what this medicine is please?', 'Represent the given image with the following question: What is the title of this book?', 'Represent the given image with the following question: Which one is the blue one?']`
- corpus samples: `['night time', 'tonic wine', 'assorted', '8:43', 'sauerkraut']`
- top corpus texts: `[('night time', 1), ('tonic wine', 1), ('assorted', 1), ('8:43', 1), ('sauerkraut', 1)]`

### ImageNet-1K
- query samples: `['Represent the given image for classification', 'Represent the given image for classification', 'Represent the given image for classification']`
- corpus samples: `['coucal', 'artichoke, globe artichoke', 'disk brake, disc brake', 'Sealyham terrier, Sealyham', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin']`
- top corpus texts: `[('coucal', 1), ('artichoke, globe artichoke', 1), ('disk brake, disc brake', 1), ('Sealyham terrier, Sealyham', 1), ('washbasin, handbasin, washbowl, lavabo, wash-hand basin', 1)]`

### SUN397
- query samples: `['Identify the scene shown in the image', 'Identify the scene shown in the image', 'Identify the scene shown in the image']`
- corpus samples: `['firing range indoor', 'raft', 'sandbox', 'waterfall fan', 'volleyball court indoor']`
- top corpus texts: `[('firing range indoor', 1), ('raft', 1), ('sandbox', 1), ('waterfall fan', 1), ('volleyball court indoor', 1)]`

### VOC2007
- query samples: `['Identify the object shown in the image', 'Identify the object shown in the image', 'Identify the object shown in the image']`
- corpus samples: `['chair', 'tvmonitor', 'sofa', 'horse', 'bicycle']`
- top corpus texts: `[('chair', 1), ('tvmonitor', 1), ('sofa', 1), ('horse', 1), ('bicycle', 1)]`

### ImageNet-A
- query samples: `['Represent the given image for classification', 'Represent the given image for classification', 'Represent the given image for classification']`
- corpus samples: `['Rottweiler', 'Sussex spaniel', 'robin, American robin, Turdus migratorius', 'artichoke, globe artichoke', 'groom, bridegroom']`
- top corpus texts: `[('Rottweiler', 1), ('Sussex spaniel', 1), ('robin, American robin, Turdus migratorius', 1), ('artichoke, globe artichoke', 1), ('groom, bridegroom', 1)]`

### ImageNet-R
- query samples: `['Represent the given image for classification', 'Represent the given image for classification', 'Represent the given image for classification']`
- corpus samples: `['flute', 'bell_pepper', 'missile', 'space_shuttle', 'African_chameleon']`
- top corpus texts: `[('flute', 1), ('bell_pepper', 1), ('missile', 1), ('space_shuttle', 1), ('African_chameleon', 1)]`

### ObjectNet
- query samples: `['Identify the object shown in the image', 'Identify the object shown in the image', 'Identify the object shown in the image']`
- corpus samples: `['tray', 'bath towel', 'paper towel', 'hammer', 'portable heater']`
- top corpus texts: `[('tray', 1), ('bath towel', 1), ('paper towel', 1), ('hammer', 1), ('portable heater', 1)]`
