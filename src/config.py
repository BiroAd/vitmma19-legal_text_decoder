"""
Configuration file for model hyperparameters and training settings.
"""

from pathlib import Path
import torch


class Config:
    """Central configuration for all models and training."""
    
    # Directory paths
    OUTPUT_DIR = Path('output')
    DATA_DIR = Path('data')
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === Baseline Model Config ===
    # Vectorizer settings
    BASELINE_MAX_FEATURES = 10000
    BASELINE_NGRAM_RANGE = (1, 2)  # unigrams and bigrams
    BASELINE_MIN_DF = 2  # minimum document frequency
    
    # Logistic Regression settings
    BASELINE_MAX_ITER = 2000
    BASELINE_CLASS_WEIGHT = 'balanced'
    BASELINE_SOLVER = 'lbfgs'
    BASELINE_RANDOM_STATE = 42
    
    # Paths (use proper path concatenation - no leading slashes!)
    BASELINE_MODEL_PATH = OUTPUT_DIR / 'model' / 'baseline' / 'baseline_model.pkl'
    VECTORIZER_PATH = OUTPUT_DIR / 'model' / 'baseline' / 'vectorizer.pkl'
    
    # === MLP Model Config ===
    # Model architecture
    MLP_SENTENCE_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    MLP_HIDDEN_DIMS = [512, 128]
    MLP_NUM_CLASSES = 5
    MLP_DROPOUT = 0.3
    MLP_USE_ORDINAL_LOSS = True
    
    # Training hyperparameters
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 100
    MLP_LEARNING_RATE = 0.002
    MLP_USE_CLASS_WEIGHTS = True
    
    # Early stopping
    MLP_EARLY_STOPPING = True
    MLP_EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
    
    # Scheduler
    MLP_SCHEDULER_FACTOR = 0.5
    MLP_SCHEDULER_PATIENCE = 3
    
    # Paths
    MLP_MODEL_PATH = OUTPUT_DIR / 'model' / 'mlp' / 'mlp_model_best.pth'
    MLP_FINAL_MODEL_PATH = OUTPUT_DIR / 'model' / 'mlp' / 'mlp_model_final.pth'
    MLP_EMBEDDINGS_PATH = OUTPUT_DIR / 'embeddings.pkl'
    MLP_HISTORY_PATH = OUTPUT_DIR / 'model' / 'mlp' / 'mlp_training_history.pkl'
    SENTENCE_TRANSFORMER_PATH = OUTPUT_DIR / 'model' / 'mlp' / 'sentence_transformer'
    
    # === Inference Config ===
    INFERENCE_OUTPUT_PATH = OUTPUT_DIR / 'inference_predictions.csv'
    
    # Sample texts for inference
    SAMPLE_TEXTS = [
        "Üzemeltető: Az, aki a Banktól függetlenül biztosítja a szolgáltatás eléréséhez szükséges, Hirdetményben meghatározott eszközöket.",
        "5.6 Az e fejezetben előírtak alkalmazandók abban az esetben is, ha az Ügyfél/Felhasználó a szolgáltatást nem az annak igénybevételéhez szükséges banki eszközökkel használja.",
        "Törzsrész: az egyes Előfizetői szolgáltatásokra vonatkozó általános jellegű rendelkezéseket tartalmazza; sz. melléklet: Jogvita esetén eljáró hatóságok, szervek megnevezése, elérhetősége sz. melléklet: Díjak (értékesíthető és lezárt díjcsomagok) sz. melléklet: Szolgáltatáscsomagok, közös kedvezmények, integrált ajánlatok sz. melléklet: Akciók részletes leírása, időtartama, feltételei, díjai és a nyújtott kedvezmények A mellékletek a jelen ÁSZF szerves részét képezik, így a törzsrész és a mellékletek együttesen alkalmazandóak. Az egyes mellékletek és a törzsrész közötti eltérés esetén a melléklet irányadó. Az egyes Előfizetői szolgáltatásokra a vonatkozó mellékletek értelemszerűen vonatkoznak. Az ÁSZF-ben, az Egyedi előfizetői szerződésben és ahhoz kapcsolódó módosításban, nyilatkozatban hivatkozott jogszabályok rövidítése Akr.: a nyilvános elektronikus hírközlési szolgáltatáshoz kapcsolódó adatvédelmi és titoktartási kötelezettségre, az adatkezelés és a titokvédelem különleges feltételeire, a hálózatok és a szolgáltatások biztonságára és integritására, a forgalmi és számlázási adatok kezelésére, valamint az azonosítókijelzésre és hívásátirányításra vonatkozó szabályokról szóló 4/2012. (I.24.) NMHH rendelet Aktv.: a termékekre és a szolgáltatásokra vonatkozó akadálymentességi követelményeknek való megfelelés általános szabályairól szóló 2022. évi XVII. törvény Eht.: az elektronikus hírközlésről szóló 2003. évi C. törvény Eszr.: az elektronikus hírközlési előfizetői szerződések részletes szabályairól szóló 22/2020. (XII.21.) NMHH rendelet Eszmr.: az elektronikus hírközlési szolgáltatás minőségének az előfizetők és felhasználók védelmével összefüggő követelményeiről, valamint a díjazás hitelességéről szóló 13/2011. (XII. 27.) NMHH rendelet Fgytv.: a fogyasztóvédelemről szóló 1997. évi CLV. törvény Gdpr.: az Európai Parlament és a Tanács 2016. április 27-i (EU) 2016/679 rendelete a természetes személyeknek a személyes adatok kezelése tekintetében történő védelméről és az ilyen adatok szabad áramlásáról, valamint a 95/46/EK irányelv hatályon kívül helyezéséről (általános adatvédelmi rendelet)",
        "Jelenlévők között az Előfizetői szerződés akkor jön létre, amikor az Előzetes tájékoztatás és az Előfizetői szerződés adatainak összefoglalója Előfizető rendelkezésére bocsátását követően a Felek az Egyedi előfizetői szerződést aláírják, vagy a szerződés megkötésére irányuló akaratukat kifejezik. Az aláírás történhet hagyományos módon, papíron tollal, vagy biztonságos biometrikus aláíráson alapuló elektronikus aláírással, amely a a Szolgáltató által biztosított elektronikus eszközön történik (így különösen PC aláíró pad vagy tablet segítségével). Üzlethelyiségen kívül az Előfizetői szerződés akkor jön létre, amikor amikor az Előzetes tájékoztatás és az Előfizetői szerződés adatainak összefoglalója Előfizető rendelkezésére bocsátását követően az a) pontban szerinti eszközök útján a Felek az Előfizetői szerződést aláírják vagy a szerződés megkötésére irányuló akaratukat kifejezik. Szóban távollévők között (telefonon) tett szerződéses jognyilatkozat esetén az Előfizetői szerződés akkor jön létre, ha az Igény szükséges adatai a Szolgáltató rendelkezésére állnak, az igényelt szolgáltatás létesíthető, és az Előfizető az Előzetes tájékoztatásban és Szerződés adatainak összefoglalójában foglalt szerződéses feltételek rendelkezésére bocsátását követően az Előfizetői szerződés megkötésére irányuló akaratát szóban kifejezte. A ráutaló magatartással létrejött szerződés létrejöttének időpontja Ráutaló magatartással akkor jön létre az Előfizetői szerződés, ha az Igény szükséges adatai a Szolgáltató rendelkezésére állnak, az igényelt szolgáltatás létesíthető, és az Előfizető az Előzetes tájékoztatásban és Szerződés adatainak összefoglalójában foglalt szerződéses feltételek rendelkezésére bocsátását követően az Előfizetői szerződés megkötésére irányuló akaratát ráutaló magatartással fejezi ki. Az Előfizetői szerződés létrejöttének napja a ráutaló magatartás tanúsításának napja. a Szolgáltatás igénybevételéhez szükséges eszközök Szolgáltatótól történő átvétele; a Szolgáltatás létesítésének lehetővé tétele és/vagy a létesítést igazoló szerelési lap aláírása; egyenleg feltöltése előre fizetett Előfizetői szolgáltatások esetén; a Szolgáltatás igénybevételi lehetőségének más felhasználó számára történő biztosítása; Előfizetői minőségben történő rendelkezés a Szolgáltatással kapcsolatban. Az Előfizetői szerződés szóban vagy ráutaló magatartással történő létrejötte esetén az Előfizetői szerződés megkötését, vagy az Előfizetői szerződés adatainak összefoglalója rendelkezésre bocsátása után a szerződés hatálybalépését követően a Szolgáltató legfeljebb 8 napon belül az Egyedi előfizetői szerződést - az előfizetői szerződés részét képező előfizetői szerződés adatainak összefoglalója kivételével - átadja az előfizetőnek, ha arra az Előfizetői szerződés megkötését vagy hatálybalépését megelőzően nem került sor. Az Egyedi előfizetői szerződést a Szolgáltató Internet-hozzáférés szolgáltatásra kötött szerződés kivételével a Felek megegyezése szerinti Tartós adathordozón, megegyezés hiányában papíron, nyomtatott formában bocsátja az Előfizető rendelkezésére. Internet-hozzáférés szolgáltatásra kötött szerződés esetében az Egyedi előfizetői szerződés rendelkezésére bocsátása Tartós adathordozón történik. A szerződéskötésre és -módosításra irányuló igény elutasítása",
        "Adatbeviteli hibáknak a szerződéses nyilatkozat elküldését megelőzően történő azonosításához és kijavításához biztosított eszközökkel kapcsolatos tájékoztatás:"
    ]


# Create a singleton instance
config = Config()
