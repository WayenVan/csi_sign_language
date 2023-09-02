import sys
sys.path.append('src')

from csi_sign_language.utils.logger import build_logger


logger = build_logger('test', 'dataset/test.log')
logger.info('hohohohhohahahahah')
logger.warn('hahsdhfh')