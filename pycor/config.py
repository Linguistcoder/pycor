import os

basename = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    DEBUG = False
    TESTING = False
    DanNet_PATH = 'H:/CST_COR/data_modeller/DanNet'
    word2vec_PATH = 'H:/CST_COR/data_modeller/word2vec/dsl_skipgram_2020_m5_f500_epoch2_w5.model.txtvectors'


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    ENV = 'development'
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
