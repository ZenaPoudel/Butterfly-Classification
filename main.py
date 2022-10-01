from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from utils import read_and_resize 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./models/mobileNetV2_NN.hdf5")

CLASS_NAMES = [
    'ADONIS',
    'AFRICAN GIANT SWALLOWTAIL',
    'AMERICAN SNOOT',
    'AN 88',
    'APPOLLO',
    'ATALA',
    'BANDED ORANGE HELICONIAN',
    'BANDED PEACOCK',
    'BECKERS WHITE',
    'BLACK HAIRSTREAK',
    'BLUE MORPHO',
    'BLUE SPOTTED CROW',
    'BROWN SIPROETA',
    'CABBAGE WHITE',
    'CAIRNS BIRDWING',
    'CHECQUERED SKIPPER',
    'CHESTNUT',
    'CLEOPATRA',
    'CLODIUS PARNASSIAN',
    'CLOUDED SULPHUR',
    'COMMON BANDED AWL',
    'COMMON WOOD-NYMPH',
    'COPPER TAIL',
    'CRECENT',
    'CRIMSON PATCH',
    'DANAID EGGFLY',
    'EASTERN COMA',
    'EASTERN DAPPLE WHITE',
    'EASTERN PINE ELFIN',
    'ELBOWED PIERROT',
    'GOLD BANDED',
    'GREAT EGGFLY',
    'GREAT JAY',
    'GREEN CELLED CATTLEHEART',
    'GREY HAIRSTREAK',
    'INDRA SWALLOW',
    'IPHICLUS SISTER',
    'JULIA',
    'LARGE MARBLE',
    'MALACHITE',
    'MANGROVE SKIPPER',
    'MESTRA',
    'METALMARK',
    'MILBERTS TORTOISESHELL',
    'MONARCH',
    'MOURNING CLOAK',
    'ORANGE OAKLEAF',
    'ORANGE TIP',
    'ORCHARD SWALLOW',
    'PAINTED LADY',
    'PAPER KITE',
    'PEACOCK',
    'PINE WHITE',
    'PIPEVINE SWALLOW',
    'POPINJAY',
    'PURPLE HAIRSTREAK',
    'PURPLISH COPPER',
    'QUESTION MARK',
    'RED ADMIRAL',
    'RED CRACKER',
    'RED POSTMAN',
    'RED SPOTTED PURPLE',
    'SCARCE SWALLOW',
    'SILVER SPOT SKIPPER',
    'SLEEPY ORANGE',
    'SOOTYWING',
    'SOUTHERN DOGFACE',
    'STRAITED QUEEN',
    'TROPICAL LEAFWING',
    'TWO BARRED FLASHER',
    'ULYSES',
    'VICEROY',
    'WOOD SATYR',
    'YELLOW SWALLOW TAIL',
    'ZEBRA LONG WING'
]


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image = read_and_resize(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)