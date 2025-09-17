PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']



OFFICEHOME_CLASSES = ["Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", "Calculator",
                      "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch", "Curtains", "Desk_Lamp",
                      "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet", "Flipflops", "Flowers", "Folder",
                      "Fork", "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard", "Knives", "Lamp_Shade",
                      "Laptop", "Marker", "Monitor", "Mop", "Mouse", "Mug", "Notebook", "Oven", "Pan", "Paper_Clip",
                      "Pen", "Pencil", "Postit_Notes", "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler",
                      "Scissors", "Screwdriver", "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "Table",
                      "Telephone", "ToothBrush", "Toys", "Trash_Can", "TV", "Webcam"]

OFFICEHOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']


VLCS_CLASSES = ["bird", "car", "chair", "dog", "person"]

VLCS_DOMAINS = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']

DOMAINNET_CLASSES = [
    # Furniture
    "bathtub", "bed", "bench", "ceiling_fan", "chair", "chandelier", "couch", "door", "dresser",
    "fence", "fireplace", "floor_lamp", "hot_tub", "ladder", "lantern", "mailbox", "picture_frame",
    "pillow", "postcard", "see_saw", "sink", "sleeping_bag", "stairs", "stove", "streetlight",
    "suitcase", "swing_set", "table", "teapot", "toilet", "toothbrush", "toothpaste", "umbrella",
    "vase", "wine_glass",
    # Mammal
    "bat", "bear", "camel", "cat", "cow", "dog", "dolphin", "elephant", "giraffe", "hedgehog",
    "horse", "kangaroo", "lion", "monkey", "mouse", "panda", "pig", "rabbit", "raccoon",
    "rhinoceros", "sheep", "squirrel", "tiger", "whale", "zebra",
    # Tool
    "anvil", "axe", "bandage", "basket", "boomerang", "bottlecap", "broom", "bucket", "compass",
    "drill", "dumbbell", "hammer", "key", "nail", "paint_can", "passport", "pliers", "rake",
    "rifle", "saw", "screwdriver", "shovel", "skateboard", "stethoscope", "stitches", "sword",
    "syringe", "wheel",
    # Cloth
    "belt", "bowtie", "bracelet", "camouflage", "crown", "diamond", "eyeglasses", "flip_flops",
    "hat", "helmet", "jacket", "lipstick", "necklace", "pants", "purse", "rollerskates", "shoe",
    "shorts", "sock", "sweater", "t-shirt", "underwear", "wristwatch",
    # Electricity
    "calculator", "camera", "cell_phone", "computer", "cooler", "dishwasher", "fan", "flashlight",
    "headphones", "keyboard", "laptop", "light_bulb", "megaphone", "microphone", "microwave",
    "oven", "power_outlet", "radio", "remote_control", "spreadsheet", "stereo", "telephone",
    "television", "toaster", "washing_machine",
    # Building
    "The_Eiffel_Tower", "The_Great_Wall", "barn", "bridge", "castle", "church", "diving_board",
    "garden", "garden_hose", "golf_club", "hospital", "house", "jail", "lighthouse", "pond",
    "pool", "skyscraper", "square", "tent", "waterslide", "windmill",
    # Office
    "alarm_clock", "backpack", "bandage", "binoculars", "book", "calendar", "candle", "clock",
    "coffee_cup", "crayon", "cup", "envelope", "eraser", "map", "marker", "mug", "nail",
    "paintbrush", "paper_clip", "pencil", "scissors",
    # Human Body
    "arm", "beard", "brain", "ear", "elbow", "eye", "face", "finger", "foot", "goatee", "hand",
    "knee", "leg", "moustache", "mouth", "nose", "skull", "smiley_face", "toe", "tooth",
    # Road Transportation
    "ambulance", "bicycle", "bulldozer", "bus", "car", "firetruck", "motorbike", "pickup_truck",
    "police_car", "roller_coaster", "school_bus", "tractor", "train", "truck", "van",
    # Food
    "birthday_cake", "bread", "cake", "cookie", "donut", "hamburger", "hot_dog", "ice_cream",
    "lollipop", "peanut", "pizza", "popsicle", "sandwich", "steak",
    # Nature
    "beach", "cloud", "hurricane", "lightning", "moon", "mountain", "ocean", "rain", "rainbow",
    "river", "snowflake", "star", "sun", "tornado",
    # Cold Blooded
    "crab", "crocodile", "fish", "frog", "lobster", "octopus", "scorpion", "sea_turtle", "shark",
    "snail", "snake", "spider",
    # Music
    "cello", "clarinet", "drums", "guitar", "harp", "piano", "saxophone", "trombone", "trumpet",
    "violin",
    # Fruit
    "apple", "banana", "blackberry", "blueberry", "grapes", "pear", "pineapple", "strawberry",
    "watermelon",
    # Sport
    "baseball", "baseball_bat", "basketball", "flying_saucer", "hockey_puck", "hockey_stick",
    "snorkel", "soccer_ball", "tennis_racquet", "yoga",
    # Tree
    "bush", "cactus", "flower", "grass", "house_plant", "leaf", "palm_tree", "tree",
    # Bird
    "bird", "duck", "flamingo", "owl", "parrot", "penguin", "swan",
    # Vegetable
    "asparagus", "broccoli", "carrot", "mushroom", "onion", "peas", "potato", "string_bean",
    # Shape
    "circle", "hexagon", "line", "octagon", "squiggle", "triangle", "zigzag",
    # Kitchen
    "fork", "frying_pan", "hourglass", "knife", "lighter", "matches", "spoon", "wine_bottle",
    # Water Transportation
    "aircraft_carrier", "canoe", "cruise_ship", "sailboat", "speedboat", "submarine",
    # Sky Transportation
    "airplane", "helicopter", "hot_air_balloon", "parachute",
    # Insect
    "ant", "bee", "butterfly", "mosquito",
    # Others
    "The_Mona_Lisa", "angel", "animal_migration", "campfire", "cannon", "dragon", "feather",
    "fire_hydrant", "mermaid", "snowman", "stop_sign", "teddy-bear", "traffic_light"
]

DOMAINNET_DOMAINS = ['real', 'clipart', 'infograph', 'painting', 'quickdraw', 'sketch']

