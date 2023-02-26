
from tfdbnet.model import DBNet
from tfdbnet.processor import PostProcessor
from tensorflow.keras.optimizers import Adam
from tfdbnet.losses import DBLoss 
from tensorflow.keras.callbacks import EarlyStopping 
from tfdbnet.metrics import TedEvalMetric 
from tfdbnet.loader import AnnotationsImporter, DataGenerator 
from tqdm.notebook import tqdm 
from time import time
from keras.callbacks import CSVLogger


class Trainer:

    def __init__(self,TRAIN_PATHS_MAP,VALIDATE_PATHS_MAP):

        self.csv_log = CSVLogger('log.csv',append = True, separator = ';')
        self.IGNORE_TEXTS = ['###']
        self.IMAGE_SIZE = 640 # Must be divisible by 32
        self.THRESH_MIN = 0.3
        self.THRESH_MAX = 0.7
        self.SHRINK_RATIO = 0.4
        self.MIN_BOX_SCORE = 0.5
        self.MAX_CANDIDATES = 1000
        self.IMAGE_SHORT_SIDE = 736 # Must be divisible by 32
        self.AREA_PRECISION_CONSTRAINT = 0.4
        self.AREA_RECALL_CONSTRAINT = 0.4
        self.TRAIN_BATCH_SIZE = 64
        self.VALIDATE_BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-3
        self.EPOCHS = 100

        self.TRAIN_PATHS_MAP = TRAIN_PATHS_MAP
        self.VALIDATE_PATHS_MAP = VALIDATE_PATHS_MAP


    def train(self):

        train_annotations = AnnotationsImporter(self.TRAIN_PATHS_MAP)
        validate_annotations = AnnotationsImporter(self.VALIDATE_PATHS_MAP) 

        train_generator = DataGenerator(
                                        train_annotations.annotations, 
                                        self.TRAIN_BATCH_SIZE, self.IMAGE_SIZE, self.IGNORE_TEXTS,  
                                        self.THRESH_MIN, self.THRESH_MAX, self.SHRINK_RATIO, seed=2022
                                        )

        validate_generator = DataGenerator(
                                        validate_annotations.annotations, 
                                        self.VALIDATE_BATCH_SIZE, self.IMAGE_SIZE, self.IGNORE_TEXTS, 
                                        self.THRESH_MIN, self.THRESH_MAX,self.SHRINK_RATIO, seed=None # No shuffle
                                            )


        post_processor = PostProcessor(min_box_score=self.MIN_BOX_SCORE, max_candidates=self.MAX_CANDIDATES)
        dbnet = DBNet(post_processor, backbone='ResNet18', k=50)

        dbnet.compile(
                optimizer = Adam(learning_rate = self.LEARNING_RATE,amsgrad=True),
                loss = DBLoss(alpha=5.0, beta=10.0, negative_ratio=3.0)
        )

        # Stop if no improvement after 10 epochs
        early_stopping_callback = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)


        # Calculate TedEvalMetric after 10 epochs
        tedeval_callback = TedEvalMetric(
            true_annotations = validate_annotations.annotations, 
            ignore_texts = self.IGNORE_TEXTS, 
            min_box_score = self.MIN_BOX_SCORE,
            image_short_side = self.IMAGE_SHORT_SIDE,
            area_precision_constraint = self.AREA_PRECISION_CONSTRAINT,
            area_recall_constraint = self.AREA_RECALL_CONSTRAINT,
            progressbar = tqdm,
            eval_best_weights = True, # Restore the best model and evaluate it on train end 
            eval_steps = 10,
        )

        t1 = time()
        history = dbnet.fit(
            train_generator,
            validation_data = validate_generator,
            validation_steps = len(validate_generator),
            steps_per_epoch = len(train_generator),
            epochs = self.EPOCHS,
            callbacks = [early_stopping_callback, tedeval_callback, self.csv_log],
            verbose = 1
        ).history
        dbnet.model.save_weights('dbnet.h5')
        t2 = time()
        print(f"Time taken for training: {t1-t2}")



TRAIN_PATHS_MAP = 'datasets/train.txt'
VALIDATE_PATHS_MAP= 'datasets/validate.txt'

trainer=Trainer(TRAIN_PATHS_MAP,VALIDATE_PATHS_MAP)
trainer.train()
print("Training Completed")